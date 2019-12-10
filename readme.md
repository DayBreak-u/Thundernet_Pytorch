# Thundernet_Pytorch
I'll push the snet146 imagenet prepare weight sonn

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

## Test
```sh
cd script
sh  test49.sh
sh  test146.sh
sh  test535.sh
```


## TODO LIST
 
 - add voc test result
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





