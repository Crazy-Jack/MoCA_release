# MoCA
Memory Concept Attention


This repo contains the MoCA module for Fast-GAN. 

## Train with MoCA
Bash command script is provided in the scripts folder to train with MoCA. Run the following command:
```
cd scripts/CUB
bash moca_64.sh
```

Currently, only single GPU is supported, however, using FastGAN backbone, we can train GAN with MoCA within 16 hours. 


### Remove noise
Use the flag `--nonoise` in the script to remove the noise injection in the generator.

### Remove momentum
Change the value of the flag `--cp_phi_momentum ` to 0 to see how the momentum can affect the performance. 

### Remove Clusters
Set the flag `--cp_num_k` to 1 to see the effect of no hierarchical arrangement of the memory. 

## Sampling and Measure FID
Use the `eval.py` to sample, for example:
```
python ../../eval.py --n_sample 7500 \
--im_size 256 \
--batch 16 \
--start_iter 10 \
--end_iter 10 \
--dist eval_results/moca_64 \
--attention_info moca_64 \
--ckpt train_results/moca_64 \
--cp_phi_momentum 0.60 \
--cp_momentum 1. \
--cp_num_k 10 \
--cp_pool_size_per_cluster 100 \
--cp_warmup_total_iter 2000 \
```

Use the `fid.py` file for measuring the FID score, e.g.
```
path_a="few-shot-images/100-shot-grumpy_cat"
python benchmarking/fid.py --batch 64 \
--size 256 \
--path_a $path_a \
--path_b eval_results/moca_64 \
--iter 10 \
--end 10 \
```

## Robustness test:
To test the robustness benefit MoCA can bring, we provide scripts to inject perturbation during inference on the model trained without noise injection. To ensure no noise is used during training, use `--nonoise` flag as shown in `scipts/CUB_nonoise/moca_64_nonoise.sh`. After training, use `scipts/CUB_nonoise/eval_moca_inject_noise.sh` ro inject noise during evaluation and measure the performane.


