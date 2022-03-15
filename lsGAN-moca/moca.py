''' Layers
    This file contains various layers for the BigGAN models.
'''
import functools

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
from torch.autograd import Variable
import faiss

from sklearn.cluster import KMeans


class MomemtumConceptAttentionProto(nn.Module):
    """concept attention"""

    def __init__(self, ch, pool_size_per_cluster, num_k, feature_dim, warmup_total_iter=1000,
                 cp_momentum=1, which_conv=nn.Conv2d,
                 cp_phi_momentum=0.6, device='cuda', use_sa=True):
        super(MomemtumConceptAttentionProto, self).__init__()
        self.myid = "atten_concept_prototypes"
        self.device = device
        self.pool_size_per_cluster = pool_size_per_cluster
        self.num_k = num_k
        self.feature_dim = feature_dim
        self.ch = ch  # input channel
        self.total_pool_size = self.num_k * self.pool_size_per_cluster

        self.register_buffer('concept_pool', torch.rand(self.feature_dim, self.total_pool_size))
        self.register_buffer('concept_proto', torch.rand(self.feature_dim, self.num_k))
        # concept pool is arranged as memory cell, i.e. linearly arranged as a 2D tensor, use get_cluster_ptr to get starting pointer for each cluster

        # states that indicating the warmup
        self.register_buffer('warmup_iter_counter', torch.FloatTensor([0.]))
        self.warmup_total_iter = warmup_total_iter
        self.register_buffer('pool_structured', torch.FloatTensor(
            [0.]))  # 0 means pool is un clustered, 1 mean pool is structured as clusters arrays

        # register attention module
        self.which_conv = which_conv
        self.theta = self.which_conv(
            self.ch, self.feature_dim, kernel_size=1, padding=0, bias=False)

        self.phi = self.which_conv(
            self.ch, self.feature_dim, kernel_size=1, padding=0, bias=False)

        self.phi_k = [self.which_conv(
            self.ch, self.feature_dim, kernel_size=1, padding=0,
            bias=False).cuda()]  # using list to prevent pytorch consider phi_k as a parameter to optimize

        for param_phi, param_phi_k in zip(self.phi.parameters(), self.phi_k[0].parameters()):
            param_phi_k.data.copy_(param_phi.data)  # initialize
            param_phi_k.requires_grad = False  # not update by gradient

        self.g = self.which_conv(
            self.ch, self.feature_dim, kernel_size=1, padding=0, bias=False)
        self.o = self.which_conv(
            self.feature_dim, self.ch, kernel_size=1, padding=0, bias=False)
        # Learnable gain parameter
        self.gamma = P(torch.tensor(0.), requires_grad=True)

        # self.momentum
        self.cp_momentum = cp_momentum
        self.cp_phi_momentum = cp_phi_momentum

        # self attention
        self.use_sa = use_sa

        # reclustering
        self.re_clustering_iters = 100
        self.clustering_counter = 0

    #############################
    #       Pool operation      #
    #############################
    @torch.no_grad()
    def _update_pool(self, index, content):
        """update concept pool according to the content
        index: [m, ]
        content: [c, m]
        """
        assert len(index.shape) == 1
        assert content.shape[1] == index.shape[0]
        assert content.shape[0] == self.feature_dim

        # print("Updating concept pool...")
        self.concept_pool[:, index] = content.clone()

    @torch.no_grad()
    def _update_prototypes(self, index, content):
        assert len(index.shape) == 1
        assert content.shape[1] == index.shape[0]
        assert content.shape[0] == self.feature_dim
        # print("Updating prototypes...")
        self.concept_proto[:, index] = content.clone()

    @torch.no_grad()
    def computate_prototypes(self):
        """compute prototypes based on current pool"""
        assert not self._get_warmup_state(), f"still in warm up state {self.warmup_state}, computing prototypes is forbidden"
        self.concept_proto = self.concept_pool.detach().clone().reshape(self.feature_dim, self.num_k,
                                                                        self.pool_size_per_cluster).mean(2)

    @torch.no_grad()
    def forward_update_pool(self, activation, cluster_num, momentum=None):
        """update activation into the concept pool after warmup in each forward pass
        activation: [m, c]
        cluster_num: [m, ]
        momentum: None or a float scalar
        """

        if not momentum:
            momentum = 1.

        # generate update index
        assert cluster_num.max() < self.num_k
        # index the starting pointer of each cluster add a rand num
        index = cluster_num * self.pool_size_per_cluster + torch.randint(self.pool_size_per_cluster,
                                                                         size=(cluster_num.shape[0],)).to(self.device)

        # adding momentum to activation
        self.concept_pool[:, index] = (1. - momentum) * self.concept_pool[:,
                                                        index].clone() + momentum * activation.detach().T

    #############################
    # Initialization and warmup #
    #############################
    @torch.no_grad()
    def pool_kmean_init_gpu(self, seed=0, gpu_num=0, temperature=1):
        """TODO: clear up
        perform kmeans for cluster concept pool initialization
        Args:
            x: data to be clustered
        """

        print('performing kmeans clustering')
        results = {'im2cluster': [], 'centroids': [], 'density': []}
        x = self.concept_pool.clone().cpu().numpy().T
        x = np.ascontiguousarray(x)
        num_cluster = self.num_k
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 100
        clus.nredo = 10
        clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = gpu_num
        index = faiss.GpuIndexFlatL2(res, d, cfg)

        clus.train(x, index)

        D, I = index.search(x, 1)  # for each sample, find cluster distance and assignments
        im2cluster = [int(n[0]) for n in I]

        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

        # sample-to-centroid distances for each cluster
        Dcluster = [[] for c in range(k)]
        for im, i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])

        # concentration estimation (phi)
        density = np.zeros(k)
        for i, dist in enumerate(Dcluster):
            if len(dist) > 1:
                d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
                density[i] = d

                # if cluster only has one point, use the max to estimate its concentration
        dmax = density.max()
        for i, dist in enumerate(Dcluster):
            if len(dist) <= 1:
                density[i] = dmax

        density = density.clip(np.percentile(density, 10),
                               np.percentile(density, 90))  # clamp extreme values for stability
        print(density.mean())
        density = temperature * density / density.mean()  # scale the mean to temperature

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids)
        centroids = nn.functional.normalize(centroids, p=2, dim=1)

        im2cluster = torch.LongTensor(im2cluster)
        density = torch.Tensor(density)

        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)

        del cfg, res, index, clus

        # rearrange
        self.structure_memory_bank(results)
        print("Finish kmean init...")
        del results

    @torch.no_grad()
    def pool_kmean_init(self, seed=0, gpu_num=0, temperature=1):
        """TODO: clear up
        perform kmeans for cluster concept pool initialization
        Args:
            x: data to be clustered
        """

        print('performing kmeans clustering')
        results = {'im2cluster': [], 'centroids': [], 'density': []}
        x = self.concept_pool.clone().cpu().numpy().T
        x = np.ascontiguousarray(x)
        num_cluster = self.num_k

        kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(x)

        centroids = torch.Tensor(kmeans.cluster_centers_)
        centroids = nn.functional.normalize(centroids, p=2, dim=1)
        im2cluster = torch.LongTensor(kmeans.labels_)

        results['centroids'].append(centroids)
        results['im2cluster'].append(im2cluster)

        # rearrange
        self.structure_memory_bank(results)
        print("Finish kmean init...")

    @torch.no_grad()
    def structure_memory_bank(self, cluster_results):
        """make memory bank structured """
        centeriod = cluster_results['centroids'][0]  # [num_k, feature_dim]
        cluster_assignment = cluster_results['im2cluster'][0]  # [total_pool_size,]

        mem_index = torch.zeros(
            self.total_pool_size).long()  # array of memory index that contains instructions of how to rearange the memory into structured clusters
        memory_states = torch.zeros(self.num_k, ).long()  # 0 indicate the cluster has not finished structured
        memory_cluster_insert_ptr = torch.zeros(self.num_k, ).long()  # ptr to each cluster block

        # loop through every cluster assignment to populate the concept pool for each cluster seperately
        for idx, i in enumerate(cluster_assignment):
            cluster_num = i
            if memory_states[cluster_num] == 0:

                # manipulating the index for populating memory
                mem_index[cluster_num * self.pool_size_per_cluster + memory_cluster_insert_ptr[cluster_num]] = idx

                memory_cluster_insert_ptr[cluster_num] += 1
                if memory_cluster_insert_ptr[cluster_num] == self.pool_size_per_cluster:
                    memory_states[cluster_num] = 1 - memory_states[cluster_num]
            else:
                # check if the ptr for this class is set to the last point
                assert memory_cluster_insert_ptr[cluster_num] == self.pool_size_per_cluster

        # what if some cluster didn't get populated enough? -- replicate
        not_fill_cluster = torch.where(memory_states == 0)[0]
        #print(f"memory_states {memory_states}")
        #print(f"memory_cluster_insert_ptr {memory_cluster_insert_ptr}")
        for unfill_cluster in not_fill_cluster:
            cluster_ptr = memory_cluster_insert_ptr[unfill_cluster]
            assert cluster_ptr != 0, f"cluster_ptr {cluster_ptr} is zero!!!"
            existed_index = mem_index[
                            unfill_cluster * self.pool_size_per_cluster: unfill_cluster * self.pool_size_per_cluster + cluster_ptr]
            #print(f"existed_index {existed_index}")
            #print(f"cluster_ptr {cluster_ptr}")
            #print(f"(self.pool_size_per_cluster {self.pool_size_per_cluster}")
            replicate_times = (self.pool_size_per_cluster // cluster_ptr) + 1  # with more replicate and cutoff
            #print(f"replicate_times {replicate_times}")
            replicated_index = torch.cat([existed_index for _ in range(replicate_times)])
            #print(f"replicated_index {replicated_index}")
            # permutate the replicate and select pool_size_per_cluster num of index
            replicated_index = replicated_index[torch.randperm(replicated_index.shape[0])][
                               :self.pool_size_per_cluster]  # [pool_size_per_cluster, ]
            # put it back
            assert replicated_index.shape[
                       0] == self.pool_size_per_cluster, f"replicated_index ({replicated_index.shape}) should has the same len as pool_size_per_cluster ({self.pool_size_per_cluster})"
            mem_index[unfill_cluster * self.pool_size_per_cluster: (
                                                                               unfill_cluster + 1) * self.pool_size_per_cluster] = replicated_index
            # update ptr
            memory_cluster_insert_ptr[unfill_cluster] = self.pool_size_per_cluster
            # update state
            memory_states[unfill_cluster] = 1

        assert (memory_states == 0).sum() == 0, f"memory_states has zeros: {memory_states}"
        assert (memory_cluster_insert_ptr != self.pool_size_per_cluster).sum() == 0, f"memory_cluster_insert_ptr didn't match with pool_size_per_cluster: {memory_cluster_insert_ptr}"

        # update the real pool
        self._update_pool(torch.arange(mem_index.shape[0]), self.concept_pool[:, mem_index])
        # initialize the prototype
        self._update_prototypes(torch.arange(self.num_k), centeriod.T.cuda())
        #print(f"Concept pool updated by kmeans clusters...")

    def _check_warmup_state(self):
        """check if need to switch warup_state to 0; when turn off warmup state, trigger k-means init for clustering"""
        # assert self._get_warmup_state(), "Calling _check_warmup_state when self.warmup_state is 0 (0 means not in warmup state)"

        if self.warmup_iter_counter == self.warmup_total_iter:
            # trigger kmean concept pool init
            self.pool_kmean_init()

    def warmup_sampling(self, x):
        """
        linearly sample input x to make it
        x: [n, c, h, w]"""
        n, c, h, w = x.shape
        assert self._get_warmup_state(), "calling warmup sampling when warmup state is 0"

        # evenly distributed across space
        sample_per_instance = max(int(self.total_pool_size / n), 1)

        # sample index
        index = torch.randint(h * w, size=(n, 1, sample_per_instance)).repeat(1, c, 1).to(
            self.device)  # n, c, sample_per_instance
        sampled_columns = torch.gather(x.reshape(n, c, h * w), 2, index)  # n, c, sample_per_instance
        sampled_columns = torch.transpose(sampled_columns, 1, 0).reshape(c,
                                                                         -1).contiguous()  # c, n * sample_per_instance

        # calculate percentage to populate into pool, as the later the better, use linear intepolation from 1% to 50% according to self.warmup_iter_couunter
        percentage = (self.warmup_iter_counter + 1) / self.warmup_total_iter * 0.5  # max percent is 50%
        print(f"percentage {percentage.item()}")
        sample_column_num = max(1, int(percentage * sampled_columns.shape[1]))
        sampled_columns_idx = torch.randint(sampled_columns.shape[1], size=(sample_column_num,))
        sampled_columns = sampled_columns[:, sampled_columns_idx]  # [c, sample_column_num]

        # random select pool idx to update
        update_idx = torch.randperm(self.concept_pool.shape[1])[:sample_column_num]
        self._update_pool(update_idx, sampled_columns)

        # update number
        # print(f"before self.warmup_iter_counter {self.warmup_iter_counter}")
        self.warmup_iter_counter += 1
        # print(f"after self.warmup_iter_counter {self.warmup_iter_counter}")

    #############################
    #       Forward logic       #
    #############################
    def forward(self, x, device="cuda", evaluation=False):
        # warmup
        if self._get_warmup_state():
            # print(
            #     f"Warmup state? {self._get_warmup_state()} self.warmup_iter_counter {self.warmup_iter_counter.item()} self.warmup_total_iter {self.warmup_total_iter}")
            # transform into low dimension
            theta = self.theta(x)  # [n, c, h, w]
            phi = self.phi(x)  # [n, c, h, w]
            g = self.g(x)  # [n, c, h, w]

            n, c, h, w = theta.shape

            # if still in warmup, skip attention
            self.warmup_sampling(phi)
            self._check_warmup_state()

            # normal self attention
            theta = theta.view(-1, self.feature_dim, x.shape[2] * x.shape[3])
            phi = phi.view(-1, self.feature_dim, x.shape[2] * x.shape[3])
            g = g.view(-1, self.feature_dim, x.shape[2] * x.shape[3])

            # Matmul and softmax to get attention maps
            beta = F.softmax(torch.bmm(theta.transpose(1, 2).contiguous(), phi), -1)

            # Attention map times g path
            o = self.o(torch.bmm(g, beta.transpose(1, 2).contiguous()).view(-1,
                                                                            self.feature_dim, x.shape[2], x.shape[3]))

            return self.gamma * o + x

        else:

            # transform into low dimension
            theta = self.theta(x)  # [n, c, h, w]
            phi = self.phi(x)
            g = self.g(x)  # [n, c, h, w]
            n, c, h, w = theta.shape

            # attend to concepts
            ## selecting cooresponding prototypes -> [n, h, w]
            theta = torch.transpose(torch.transpose(theta, 0, 1).reshape(c, n * h * w), 0,
                                    1).contiguous()  # n * h * w, c
            phi = torch.transpose(torch.transpose(phi, 0, 1).reshape(c, n * h * w), 0, 1).contiguous()  # n * h * w, c
            g = torch.transpose(torch.transpose(g, 0, 1).reshape(c, n * h * w), 0, 1).contiguous()  # n * h * w, c
            with torch.no_grad():
                theta_atten_proto = torch.matmul(theta, self.concept_proto.detach().clone())  # n * h * w, num_k
                cluster_affinity = F.softmax(theta_atten_proto, dim=1)  # n * h * w, num_k
                # print(f"cluster_affinity.max(1) {cluster_affinity.max(1)}")
                cluster_assignment = cluster_affinity.max(1)[1]  # [n * h * w, ]

            # for loop for each cluster
            # store mapping

            dot_product = []
            cluster_indexs = []

            for cluster in range(self.num_k):
                cluster_index = torch.where(cluster_assignment == cluster)[0]  # [n * h * w]
                theta_cluster = theta[cluster_index]  # number of data  belong to the same cluster, c

                # attend to certain cluster
                cluster_pool = self.concept_pool.detach().clone()[:, cluster * self.pool_size_per_cluster: (
                                                                                                                       cluster + 1) * self.pool_size_per_cluster]  # [c, pool_size_per_cluster]

                theta_cluster_attend_weight = torch.matmul(theta_cluster,
                                                           cluster_pool)  # [num_data_in_cluster, pool_size_per_cluster]
                # # map to back
                # beta_cluster = torch.matmul(theta_cluster_attend_weight, cluster_pool.T) # [num_data_in_cluster, c]

                dot_product.append(theta_cluster_attend_weight)
                cluster_indexs.append(cluster_index)

            # integrate into one tensor
            dot_product = torch.cat(dot_product, axis=0)  # [n * h * w, pool_size_per_cluster] but with different order
            cluster_indexs = torch.cat(cluster_indexs, axis=0)

            # remap it back into order Variable(torch.ones(2, 2), requires_grad=True)
            mapping_to_normal_index = torch.argsort(cluster_indexs)
            similarity_clusters = dot_product[mapping_to_normal_index]  # n * h * w, pool_size_per_cluster

            # dot product with context
            similarity_context = torch.bmm(theta.reshape(n, h * w, c),
                                           torch.transpose(phi.reshape(n, h * w, c), 1, 2))  # [n, h*w, h*w]
            similarity_context = similarity_context.reshape(n * h * w, h * w)  # n * h * w, h * w
            if self.use_sa:
                atten_weight = torch.cat([similarity_clusters, similarity_context],
                                         axis=1)  # [n * h * w, pool_size_per_cluster + h * w]
            else:
                atten_weight = similarity_clusters  # [n * h * w, pool_size_per_cluster]
            atten_weight = F.softmax(atten_weight, dim=1)  # [n * h * w, pool_size_per_cluster + h * w]

            # attend
            pool_residuals = []
            cluster_indexs = []
            for cluster in range(self.num_k):
                cluster_index = torch.where(cluster_assignment == cluster)[0]  # [n * h * w]
                theta_cluster = theta[cluster_index]  # number of data  belong to the same cluster, c
                atten_weight_pool_cluster = atten_weight[cluster_index,
                                            :self.pool_size_per_cluster]  # [number of data  belong to the same cluster, pool_size_per_cluster]
                # attend to certain cluster
                cluster_pool = self.concept_pool.detach().clone()[:, cluster * self.pool_size_per_cluster: (
                                                                                                                       cluster + 1) * self.pool_size_per_cluster]  # [c, pool_size_per_cluster]
                pool_residual = torch.matmul(atten_weight_pool_cluster,
                                             cluster_pool.T)  # [num_batch_data_in_cluster, c]
                pool_residuals.append(pool_residual)
                cluster_indexs.append(cluster_index)
            pool_residuals = torch.cat(pool_residuals, axis=0)  # [n * h * w, c] but with different order
            cluster_indexs = torch.cat(cluster_indexs, axis=0)

            # remap it back into order
            mapping_to_normal_index = torch.argsort(cluster_indexs)
            pool_residuals = pool_residuals[mapping_to_normal_index]  # n * h * w, c with correct order
            pool_residuals = pool_residuals.reshape(n, h * w, c)  # n, h * w, c

            # add with context
            if self.use_sa:
                atten_weight_context = atten_weight[:, self.pool_size_per_cluster:]  # [n * h * w, h * w]
                atten_weight_context = atten_weight_context.reshape(n, h * w, h * w)  # n, h*w, h*w
                context_residuals = torch.bmm(atten_weight_context, g.reshape(n, h * w,
                                                                              c))  # n, h * w, c, context residual is calcuated by g not phi
                beta_residual = pool_residuals + context_residuals  # n, h * w, c
            else:
                beta_residual = pool_residuals
            # integrate context residual with pool residual
            beta_residual = torch.transpose(beta_residual, 1, 2).reshape(n, c, h, w).contiguous()

            # print(f"beta_residual {beta_residual.shape}")
            o = self.o(beta_residual)  # n, h, w, c

            ### update pool
            with torch.no_grad():
                # moca update
                phi_k = self.phi_k[0](x)  # [n, c, h, w]
                phi_k = torch.transpose(torch.transpose(phi_k, 0, 1).reshape(c, n * h * w), 0,
                                        1).contiguous()  # n * h * w, c
                phi_k_atten_proto = torch.matmul(phi_k, self.concept_proto.detach().clone())  # n * h * w, num_k
                phi_k_atten_proto = phi_k_atten_proto.reshape(n, h * w, -1)  # n, h * w, num_k
                cluster_affinity_phi_k = F.softmax(phi_k_atten_proto, dim=2)  # n, h * w, num_k
                # print(f"cluster_affinity.max(1) {cluster_affinity.max(1)}")
                cluster_assignment_phi_k = cluster_affinity_phi_k.max(2)[1].reshape(n * h * w, )  # [n * h * w, ]

                # update pool first to allow contextual information
                # should use the lambda to update concept pool momentumlly
                self.forward_update_pool(phi_k, cluster_assignment_phi_k, momentum=self.cp_momentum)

                # update prototypes
                self.computate_prototypes()

                # update phi_k
                for param_q, param_k in zip(self.phi.parameters(), self.phi_k[0].parameters()):
                    param_k.data = param_k.data * self.cp_phi_momentum + param_q.data * (1. - self.cp_phi_momentum)

                # perform clustering again and re-assign the prototypes
                self.clustering_counter += 1
                self.clustering_counter = self.clustering_counter % self.re_clustering_iters
                # print(f"self.clustering_counter {self.clustering_counter}; self.re_clustering_iters {self.re_clustering_iters}")
                if self.clustering_counter == 0:
                    self.pool_kmean_init()

            if evaluation:
                return o * self.gamma + x, cluster_affinity

            return o * self.gamma + x

    #############################
    #     Helper  functions     #
    #############################

    def get_cluster_num_index(self, idx):
        assert idx < self.total_pool_size
        return idx // self.pool_size_per_cluster

    def get_cluster_ptr(self, cluster_num):
        """get starting pointer for cluster_num"""
        assert cluster_num < self.num_k, f"cluster_num {cluster_num} out of bound (totally has {self.num_k} clusters)"
        return self.pool_size_per_cluster * cluster_num

    def _get_warmup_state(self):
        # print(f"NOW ---- self.warmup_iter_counter {self.warmup_iter_counter}")
        return self.warmup_iter_counter.cpu() <= self.warmup_total_iter