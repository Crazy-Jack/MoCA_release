#!/bin/bash

python ../../train_noisefree.py --path CUB_200_2011/train/train_images \
--im_size 256 \
--batch_size 8 \
--iter 100000 \
--start_iter 0 \
--attention_info SA_64 \
--name baseline_SA64 \
