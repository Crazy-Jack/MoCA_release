import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import utils as vutils

import os
import random
import argparse
from tqdm import tqdm

from models import Generator


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

def resize(img):
    return F.interpolate(img, size=256)

def batch_generate(zs, netG, batch=8):
    g_images = []
    with torch.no_grad():
        for i in range(len(zs)//batch):
            g_images.append( netG(zs[i*batch:(i+1)*batch]).cpu() )
        if len(zs)%batch>0:
            g_images.append( netG(zs[-(len(zs)%batch):]).cpu() )
    return torch.cat(g_images)

def batch_save(images, folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    for i, image in enumerate(images):
        vutils.save_image(image.add(1).mul(0.5), folder_name+'/%d.jpg'%i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='generate images'
    )
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--start_iter', type=int, default=6)
    parser.add_argument('--end_iter', type=int, default=10)

    parser.add_argument('--dist', type=str, default='.')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch', default=16, type=int, help='batch size')
    parser.add_argument('--n_sample', type=int, default=2000)
    parser.add_argument('--big', action='store_true')
    parser.add_argument('--im_size', type=int, default=1024)
    # attention specific
    parser.add_argument('--attention_info', type=str, default='SA_128')
    # concept attention
    parser.add_argument("--cp_pool_size_per_cluster", type=int, default=100)
    parser.add_argument("--cp_num_k", type=int, default=10)
    parser.add_argument("--cp_feature_dim", type=int, default=128)
    parser.add_argument("--cp_warmup_total_iter", type=int, default=2000)
    parser.add_argument("--cp_momentum", type=float, default=0.6)
    parser.add_argument("--cp_phi_momentum", type=float, default=0.95)
    parser.add_argument("--cp_topk_percent", type=float, default=0.1)
    parser.add_argument("--inject_noise", type=float, default=2)
    parser.add_argument("--old_moca", action='store_true')
    parser.add_argument("--seperate_norm", default=False, action='store_true')
    parser.set_defaults(big=False)
    args = parser.parse_args()

    noise_dim = 256
    device = torch.device('cuda:%d'%(args.cuda))
    
    net_ig = Generator(args, ngf=64, nz=noise_dim, nc=3, im_size=args.im_size, noise=False)#, big=args.big )
    net_ig.to(device)

    for epoch in [10000*i for i in range(args.start_iter, args.end_iter+1)]:
        ckpt = os.path.join(args.ckpt, 'models/%d.pth'%(epoch))
        checkpoint = torch.load(ckpt, map_location=lambda a,b: a)
        net_ig.load_state_dict(checkpoint['g'])
        #load_params(net_ig, checkpoint['g_ema'])

        #net_ig.eval()
        print('load checkpoint success, epoch %d'%epoch)

        net_ig.to(device)

        del checkpoint

        dist = os.path.join(args.dist, 'eval_%d'%(epoch))
        dist = os.path.join(dist, 'img')
        os.makedirs(dist, exist_ok=True)

        # get current sample status
        existed_samples = os.listdir(dist)
        if existed_samples: 
            sampled_i = [int(str(name).split('.')[0]) for name in existed_samples]
            max_i = max(sampled_i)
            starting_batch = max_i // args.batch + 1
            print(f"Starting from {max_i}.png...")
            
        else:
            starting_batch = 0
        

        myrange = range(starting_batch, args.n_sample//args.batch)
        with torch.no_grad():
            for i in tqdm(myrange):
                noise = torch.randn(args.batch, noise_dim).to(device)
                g_imgs = net_ig(noise, inject_noise=args.inject_noise)[0]
                g_imgs = F.interpolate(g_imgs, 512)
                for j, g_img in enumerate( g_imgs ):
                    vutils.save_image(g_img.add(1).mul(0.5), 
                        os.path.join(dist, '%d.png'%(i*args.batch+j)))#, normalize=True, range=(-1,1))
