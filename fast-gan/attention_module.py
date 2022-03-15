import torch.nn as nn 
import torch
from torch.nn import Parameter as P
import torch.nn.functional as F
import moca


class SelfAttention(nn.Module):
    def __init__(self, ch, which_conv=nn.Conv2d, name='attention'):
        super(SelfAttention, self).__init__()
        self.myid = "atten"
        
        # Channel multiplier
        self.ch = ch
        print(f"INSIDE ATTENTION   self.ch // 2 {self.ch // 2}")
        self.which_conv = which_conv
        self.theta = self.which_conv(
            self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.phi = self.which_conv(
            self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.g = self.which_conv(
            self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.o = self.which_conv(
            self.ch // 8, self.ch, kernel_size=1, padding=0, bias=False)
        # Learnable gain parameter
        self.gamma = P(torch.tensor(0.), requires_grad=True)

    def forward(self, x):
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2, 2])
        g = F.max_pool2d(self.g(x), [2, 2])
        # Perform reshapes
        theta = theta.view(-1, self.ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self.ch // 8, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(-1,
                                                           self.ch // 8, x.shape[2], x.shape[3]))
        return self.gamma * o + x






def attention(mode, inchannel, resolution, args):
    if mode == 'SA':
        return SelfAttention(inchannel)
    
    elif mode == 'moca':
        
        mylayer = moca.MemoryConceptAttentionProto(
            inchannel, nn.Conv2d, args.cp_pool_size_per_cluster, 
            args.cp_num_k, args.cp_feature_dim, warmup_total_iter=args.cp_warmup_total_iter, cp_momentum=args.cp_momentum, \
            cp_phi_momentum=args.cp_phi_momentum,
        )
            
        return mylayer
    

