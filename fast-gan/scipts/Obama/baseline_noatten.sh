#!/bin/bash

python ../../train.py --path few-shot-images/100-shot-grumpy_cat \
--im_size 256 \
--batch_size 8 \
--iter 100000 \
--start_iter 0 \
--attention_info None_64 \
--name baseline_noatten \
