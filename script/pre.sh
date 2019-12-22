#!/bin/bash
set -e
cd ..

CUDA_VISIBLE_DEVICES=0  python demo.py --dataset pascal_voc_0712 --net snet_146 --load_dir snet146_2 \
       --checkepoch 6  --cuda \
        --image_dir /mnt/data1/yanghuiyu/project/object_detect/thundernetbylightheadrcnn/voc_images/input
