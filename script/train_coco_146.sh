#!/bin/bash
set -e
cd ..

CUDA_VISIBLE_DEVICES=0  python trainval_net.py --dataset coco --net snet_146 --bs 16 --nw 8 \
     --lr 1e-2   --epochs 50 --cuda  --lr_decay_step 10,30,40  --use_tfboard  True\
     --save_dir snet_146_coco  --eval_interval 1   --logdir snet146_coco_log --pre ./weights/snet_146.tar \
#     --r True --checkepoch 0
