#!/bin/bash

python ../../train_noisefree.py --path few-shot-images/100-shot-obama/img \
--im_size 256 \
--batch_size 8 \
--iter 100000 \
--start_iter 0 \
--attention_info SA_64 \
--name baseline_SA64 \
