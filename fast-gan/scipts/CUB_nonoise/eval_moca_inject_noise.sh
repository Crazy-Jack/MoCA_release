exp_name="moca_64_nonoise";
noise=0.3;
dist="eval_results/$exp_name+inject_noise=$noise";

# sample
python ../../eval_inject_noise_to_noisefree.py --n_sample 7500 \
--im_size 256 \
--batch 16 \
--start_iter 10 \
--end_iter 10 \
--dist $dist \
--ckpt train_results/$exp_name \
--inject_noise $noise \
--attention_info moca_64 \
--cp_phi_momentum 0.60 \
--cp_momentum 1. \
--cp_num_k 10 \
--cp_pool_size_per_cluster 100 \
--cp_warmup_total_iter 2000 \


path_a="CUB_200_2011/train/"
python ../../benchmarking/fid.py --batch 64 \
--size 256 \
--path_a $path_a \
--path_b $dist \
--iter 10 \
--end 10 \