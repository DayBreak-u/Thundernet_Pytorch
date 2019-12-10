#!/bin/bash
set -e
cd ..

CUDA_VISIBLE_DEVICES=0  python trainval_net.py --dataset pascal_voc_0712 --net snet_146 --bs 32 --nw 8 \
    --lr 1e-2   --epochs 150 --cuda  --lr_decay_step 50,70,100  --use_tfboard  True\
     --save_dir snet_146  --eval_interval 5   --logdir snet146_log --pre ./weights/sent_146.pth.tar \
#     --r True --checkepoch 4
