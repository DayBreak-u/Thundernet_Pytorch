import sys
import random
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from .load_voctxt import get_crowd





class VocGenerator(Dataset):
    def __init__(self, path, type, config, preloaded=False, transform=None):
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


        target = {}
        target['boxes']= gts
        target['labels'] = labels
        target['image_id'] = img_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target


