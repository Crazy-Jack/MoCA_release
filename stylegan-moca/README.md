# MoCA
Memory Concept Attention for StyleGAN-2


This repo contains the MoCA module for StyleGAN-2. The code is based on the official Pytorch implementation
of [StyleGAN2](https://github.com/NVlabs/stylegan2-ada-pytorch). For more details about StyleGAN2, please refer to
the original official repo.

## Train with MoCA
Bash command script is provided in the scripts folder to train with MoCA. 

To train a StyleGAN2 with MoCA,
Run the following command:
```
cd scripts/cifar10_moca_cluster2
bash train.sh
```

Currently only multi-GPU training with DDP mode is supported, but evaluation can be carried out with only single GPU.

### Use MoCA or not?
The flag `--use_ca_g` determines whether to add MoCA to the Generator.
The flag `--use_ca_d` determines whether to add MoCA to the Discriminator.
If not provided, fall back to original StyleGAN-2.


### How many cluster to use?
Use the flag `--cluster_num` in the script to set how many concept clusters will be used.
If set to 1, then no clustering mechanism is applied.



### How large for each concept pool(cluster)?
Use the flag `--pool_size` in the script to set the number of concepts inside a single cluster.


### Set momentum
Change the value of the flag `--momentum` to 0 to see how the momentum can affect the performance. 



