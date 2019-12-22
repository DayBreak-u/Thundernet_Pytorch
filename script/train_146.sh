#!/bin/bash
set -e
cd ..

CUDA_VISIBLE_DEVICES=0  python trainval_net.py --dataset pascal_voc_0712 --net snet_146 --bs 64 --nw 8 \
     --lr 1e-2   --epochs 150 --cuda  --lr_decay_step 25,50,75  --use_tfboard  True \
     --save_dir snet146  --eval_interval 2   --logdir snet146_log --pre ./weights/snet_146.tar \
     --r True --checkepoch 2
