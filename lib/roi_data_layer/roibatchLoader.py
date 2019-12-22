import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
from roi_data_layer.augmentation import SSDAugmentation
import model.utils.config  as  config

cfg = config.cfg

class Detection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
    """

    def __init__(self, roidb, num_classes, training=True,transform=None):
        self._roidb = roidb
        self.training = training
        self.transform = transform
        self.num_classes = num_classes
        self.max_num_box = cfg.MAX_NUM_GT_BOXES



    def __len__(self):
        return len(self._roidb)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, height, width).
                   target is the object returned by ``coco.loadAnns``.
        """

        if self.training:
            index , size = index
        else:
            size  = cfg.TEST.SIZE
        self.transform = SSDAugmentation(size, cfg.PIXEL_MEANS)
        roidb = self._roidb[index]
        im  = cv2.imread(roidb['image'])
        if len(im.shape) == 2:
            im = im[:, :, np.newaxis]
            im = np.concatenate((im, im, im), axis=2)
            # flip the channel, since the original one using cv2

        if roidb['flipped']:
            im = im[:, ::-1, :]
        height, width = im.shape[0], im.shape[1]

        boxes =  roidb['boxes']
        gt_classes = roidb['gt_classes']



        boxes_all = []
        for b,class_gt in zip(boxes,gt_classes):
            boxes_all.append([b[0]/width,b[1]/height,b[2]/width,b[3]/height,class_gt])



        target =  np.array(boxes_all)

        target_re = np.zeros([self.max_num_box,5])

        if self.transform is not None:

            img, boxes, labels = self.transform(im, target[:,:4],
                                                target[:,4])

            img = img.transpose(2, 0, 1)
            number_box  = 0
            for box in boxes:
                if number_box>=20:
                    break
                target_re[number_box] =   np.array([box[0]*size ,box[1]*size,box[2]*size,box[3]*size,labels[number_box]])

                number_box+=1
            # target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            # img_id, img, gt_boxes_padding, img_info, num_gt_boxes

        data = torch.as_tensor(img, dtype=torch.float32)
        im_info = torch.from_numpy(np.array([img.shape[1], img.shape[2],  size/width ,size/height ]))
        im_info = im_info.view(4)
        gt_boxes = torch.as_tensor(target_re, dtype=torch.int16)


        if self.training:
            return data, im_info, gt_boxes, number_box
        else:

            # im_info = np.array([[im.shape[1], im.shape[2], ratio]], dtype=np.float32)
            im_info = np.array([[img.shape[1], img.shape[2],  size/width ,size/height]], dtype=np.float32)
            im_info = torch.as_tensor(im_info, dtype=torch.float32)
            im_info = im_info.view(4)
            gt_boxes = torch.FloatTensor([1, 1, 1, 1, 1])

            return data, im_info, gt_boxes, number_box


