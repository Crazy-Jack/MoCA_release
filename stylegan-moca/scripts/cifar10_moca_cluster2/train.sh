dataset=dir_to_data;

python ../../train.py --outdir=./training-runs --data=$dataset --gpus=4 \
--kimg 16000 \
--use_ca_g True --momentum 0.999 \
--snap 200 \
--pool_size 256 \
--cluster_num 32 \
--preheat True;
