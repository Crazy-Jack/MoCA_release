#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PAT:/home/tianqinl/.conda/envs/3D_pytorch/lib/

python ../../train.py --path few-shot-images/100-shot-obama/img \
--im_size 256 \
--batch_size 8 \
--iter 100000 \
--start_iter 0 \
--attention_info None_64 \
--name baseline_noatten \
