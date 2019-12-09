#!/bin/bash
set -e
cd ..

CUDA_VISIBLE_DEVICES=0  python trainval_net.py --dataset pascal_voc_0712 --net snet_49 --bs 16 --nw 8 \
    --lr 1e-2   --epochs 150 --cuda  --lr_decay_step 50,75,100  --use_tfboard  True\
     --save_dir snet_49  --eval_interval 5   \
     --r True  --checkepoch 4
