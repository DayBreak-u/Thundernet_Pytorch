#!/bin/bash
set -e
cd ..

CUDA_VISIBLE_DEVICES=0  python trainval_net.py --dataset pascal_voc_0712 --net snet_146 --bs 8 --nw 8 \
    --lr 1e-2   --epochs 250 --cuda  --lr_decay_step 70,150,200  --use_tfboard  True\
     --save_dir snet_146_3  --eval_interval 5   --logdir snet146_3_log \
     --r True --checkepoch 4
