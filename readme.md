# Thundernet_Pytorch
## 20191222 update
- add data augmentation
- add Multi-scale training
- add onnx (doing)

## pretrained model
- train code in : https://github.com/ouyanghuiyu/Snet

## Requirements
* pytorch 1.2.0
* torchvision 0.4



## Lib Prepare 
```sh
git clone https://github.com/ouyanghuiyu/Thundernet_Pytorch.git
```

### Build  
```sh
cd lib && python setup.py  build_ext --inplace
cd psroialign/PSROIAlign && sh build.sh 
 ```   
## Data Prepare 
Download VOC0712 datasets 
ln -s "YOUR PATH" data

## Train
```sh
cd script
sh  train_49.sh
sh  train_146.sh
sh  train_535.sh
```

## demo
```sh
cd script
sh  pre.sh

```

## TODO LIST
 
 - add coco train and test
 - add NCNN inference

## Citation
Please cite the paper in your publications if it helps your research:
```
@article{zheng2019thundernet,
  title={ThunderNet: Towards Real-time Generic Object Detection},
  author={Zheng Qin, Zeming Li,Zhaoning Zhang,Yiping Bao,Gang Yu, Yuxing Peng, Jian Sun},
  journal={arXiv preprint arXiv:1903.11752},
  year={2019}
}
```

## VOC TEST EXAMPLE
![test](https://github.com/ouyanghuiyu/Thundernet_Pytorch/blob/master/voc_images/output/2008_000005.jpg)






