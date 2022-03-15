#!/bin/bash

python ../../train.py --path  CUB_200_2011/train/train_images \
--im_size 256 \
--batch_size 8 \
--iter 100000 \
--start_iter 0 \
--attention_info moca_64 \
--name moca_64 \
--cp_phi_momentum 0.60 \
--cp_momentum 1. \
--cp_num_k 10 \
--cp_pool_size_per_cluster 100 \
--cp_warmup_total_iter 2000 \
