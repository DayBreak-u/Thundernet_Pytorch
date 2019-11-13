Configs = {
    "VOCROOT" : "/mnt/data1/yanghuiyu/datas/voc0712/VOC/VOCdevkit/",
    "train_txts" : ["./data_pre/VOC2007_trainval.txt" , "./data_pre/VOC2012_trainval.txt" ] ,
    "val_txts" : ["./data_pre/VOC2007_test.txt"] ,

    "CEM_FILTER" : 5*7*7 ,
    "representation_size" : 1024 ,
    "rpn_dense" : 256 ,
    "num_classes" : 21 ,
    "Snet_version" : 146 , # 49 , 146 ,535
    "anchor_sizes" : (( 32,  64, 128,  256  , 512) ,),
    "aspect_ratios" : (( 1./2. , 3./4. , 1. ,4./3. , 2.) ,),
    "anchor_number" : 5 * 5,
    "spatial_scale" :  1./16. ,
    "Multi_size" : [ 320,320,320] ,


    "CLASSES" : ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                   'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                   'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
}
