import sys
import random
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from .load_voctxt import get_crowd

from .data_augment import  Preproc

Preproc = Preproc()



#
# class VocGenerator(Dataset):
#     def __init__(self, path, type, transform=None):
#         self.dataset = get_crowd(path, type=type)
#         self.dataset_len = len(self.dataset)
#         self.type = type
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, item):
#
#         if self.type == "train":
#
#
#             img_data = self.dataset[item]
#             # img = Image.open(img_data["filepath"])
#             img = cv2.imread(img_data["filepath"])
#
#             img_id = torch.tensor([item])
#
#             gts = img_data["bboxes"].copy()
#
#             labels = img_data["labels"].copy()
#
#             image_t, boxes_t, labels_t =  Preproc(img ,gts  , labels     )
#
#
#             debug = 0
#             if debug:
#                 image_t_ = np.swapaxes(image_t,0,2)
#                 image_t_ = np.swapaxes(image_t_,0,1)
#                 cv2.imwrite("test.jpg", image_t_)
#                 image_t_ = cv2.imread("test.jpg")
#                 for box in boxes_t:
#
#                     box = box.astype(np.int)
#                     cv2.rectangle(image_t_, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
#
#
#                 cv2.imwrite("temp.jpg", image_t_)
#
#
#         else:
#             img_data = self.dataset[item]
#             image_t = Image.open(img_data["filepath"])
#             img_id = torch.tensor([item])
#
#             boxes_t = img_data["bboxes"].copy()
#             labels_t = img_data["labels"].copy()
#
#         boxes_t = boxes_t.reshape(-1,)
#         gts_length = len(boxes_t)
#         iscrowd = torch.zeros((gts_length,), dtype=torch.int64)
#         boxes_t = torch.as_tensor(boxes_t, dtype=torch.float32)
#         labels = torch.as_tensor(labels_t, dtype=torch.int64)
#         area = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])
#
#
#         target = {}
#         target['boxes']= boxes_t
#         target['labels'] = labels
#         target['image_id'] = img_id
#         target['area'] = area
#         target['iscrowd'] = iscrowd
#
#         img  = Image.fromarray(cv2.cvtColor(image_t.astype(np.uint8), cv2.COLOR_BGR2RGB))
#
#         if self.transform is not None:
#             img, target = self.transform(img, target)
#
#         return img, target
# #
class VocGenerator(Dataset):
    def __init__(self, path, type, preloaded=False, transform=None):
        self.dataset = get_crowd(path, type=type)
        self.dataset_len = len(self.dataset)
        self.type = type
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img_data = self.dataset[item]
        img = Image.open(img_data["filepath"])
        img_id = torch.tensor([item])

        gts = img_data["bboxes"].copy()
        labels = img_data["labels"].copy()
        gts_length = len(gts)
        gts = torch.as_tensor(gts, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        area = (gts[:, 3] -gts[:, 1]) * (gts[:,2] - gts[:, 0])
        iscrowd = torch.zeros((gts_length, ), dtype=torch.int64)

        # print(labels.shape)

        target = {}
        target['boxes']= gts
        target['labels'] = labels
        target['image_id'] = img_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target


