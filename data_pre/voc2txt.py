# !/usr/bin/evn python
# coding:utf-8
import os


try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
index_map = dict(zip(CLASSES, range(len(CLASSES))))
print(index_map)

ROOT = "/mnt/data1/yanghuiyu/datas/voc0712/VOC/VOCdevkit/"




def convert_annotation(in_file,outfile):


    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text

        if cls not in CLASSES:
            print(cls)
            print(in_file)
            continue
        cls_id = index_map[cls]
        # print(cls_id)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        x1,x2,y1,y2 = b
        w ,h = x2-x1,y2-y1
        outfile.write("{} {} {} {} {}\n".format(x1,y1,w,h,cls_id))


count = 0
for ImageSet in ["VOC2007" , "VOC2012"]:
    set_root = os.path.join(ROOT,ImageSet)

    for model in ["trainval","test"]:
        txt_path = os.path.join(set_root,"ImageSets/Main/{}.txt".format(model))


        list_file = open('{}_{}.txt'.format(ImageSet,model), 'w')


        for image_id in open(txt_path):
            image_id  = image_id.strip().split()[0]
            try:
                in_file = os.path.join(set_root,"Annotations","{}.xml".format(image_id))
                list_file.write("#{}/JPEGImages/{}.jpg\n".format(ImageSet,image_id))

                convert_annotation(in_file,list_file)
            except:

                continue

        list_file.close()
