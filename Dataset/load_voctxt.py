import os
import numpy as np
from scipy import io as scio
import pandas as pd
import json
import cv2

from config import Configs


VOCROOT = Configs.get("VOCROOT")
CLASSES  = Configs.get("CLASSES")


count_class = {'person': 15576, 'chair': 4338, 'car': 4008, 'bottle': 2116, 'dog': 2079, 'bird': 1820,
               'pottedplant': 1724, 'cat': 1616, 'boat': 1397, 'sheep': 1347, 'aeroplane': 1285, 'sofa': 1211,
               'bicycle': 1208, 'tvmonitor': 1193, 'horse': 1156, 'motorbike': 1141, 'cow': 1058, 'diningtable': 1057,
               'train': 984, 'bus': 909}


def cal_repeat_time(labels):
    labels = set(labels)
    if 0 in labels:
        return 1
    min_number = np.inf
    for label in labels:
        min_number = min(min_number, count_class[CLASSES[label]])
    return count_class["person"] // min_number // 2



def get_crowd(txt_path , type="train", vis=True):

    isFirst = True
    image_dat = []
    datas = []
    for path in txt_path:
        datas.extend(open(path, "r").readlines())
    for line in datas:
        line = line.rstrip()
        if line.startswith('#'):
            if isFirst:
                boxes = []
                labels = []
                annotation = {}
                isFirst = False
            else:
                if len(boxes) == 0:
                    continue
                annotation["filepath"] = path
                annotation["bboxes"] = np.array(boxes)
                annotation["labels"] = np.array(labels)
                repeat_time = 1
                # if type == "train":
                #     repeat_time = cal_repeat_time(labels)
                for _ in range(repeat_time):
                    image_dat.append(annotation)
                boxes = []
                labels = []
                annotation = {}
            path = line[1:].strip()

            path = os.path.join(VOCROOT, path)
        else:
            line = line.split(' ')
            label = [float(x) for x in line]
            x1, y1, w, h = label[:4]
            x2 = x1 + w
            y2 = y1 + h
            class_id = label[4]

            boxes.append([int(x1),int(y1),int(x2),int(y2)])
            labels.append(int(class_id) + 1  )


    return  image_dat



#
#
#
#
# def get_crowd(root_dir="/home/princemerveil/Downloads/CrowdHuman", type="train", vis=True):
#     print("loading "  + type + " dataset")
#     imgs_paths = os.path.join(root_dir, "Images/")
#     validation_imgs = os.path.join(root_dir, "Images_validation/")
#     annotation_path = os.path.join(root_dir, "annotations")
#     train_annotation = os.path.join(annotation_path, "annotation_train.odgt")
#     validation_annotation = os.path.join(annotation_path, "annotation_val.odgt")
#
#     image_dat = []
#
#     if type == "train":
#         image_to_take = imgs_paths
#         annotation_to_take = train_annotation
#     else:
#         image_to_take = validation_imgs
#         annotation_to_take = validation_annotation
#
#
#
#     with open(annotation_to_take) as f:
#         lines = f.readlines()
#         for line in lines:
#             data = json.loads(line)
#             img_name = image_to_take + data['ID'] + '.jpg'
#             img_id = data['ID']
#             ig_boxes = []
#             boxes = []
#             vis_boxes = []
#             full_body = []
#             for box in data['gtboxes']:
#                 ignore = box["head_attr"]["ignore"] if box["tag"] == "person" and "ignore" in box["head_attr"] else 1
#                 head_box = box["hbox"]
#                 full_box = box["fbox"]
#                 visble_box = box["vbox"]
#                 if ignore == 0:
#                     boxes.append([int(visble_box[0]), int(visble_box[1]), int(visble_box[2]) + int(visble_box[0]), int(visble_box[3]) + int(visble_box[1])])
#                 else:
#                     ig_boxes.append([int(head_box[0]), int(head_box[1]), int(head_box[2]) + int(head_box[0]), int(head_box[3]) + int(head_box[1])])
#             annotation = {}
#             annotation['filepath'] = img_name
#             annotation['bboxes'] = boxes
#             annotation["vis_bboxes"] = boxes
#             annotation["ignoreareas"] = ig_boxes
#             annotation['ID'] = img_id
#             image_dat.append(annotation)
#     return  image_dat