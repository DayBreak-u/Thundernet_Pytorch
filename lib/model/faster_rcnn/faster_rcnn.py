import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torchvision.ops import RoIAlign, RoIPool

from psroialign.psroialign import PSROIAlignhandle,PSROIPoolhandle

# from psroi_pooling.modules.psroi_pool import PSRoIPool

from .modules import  RPN,SAM
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.loss.losses import _smooth_l1_loss

from model.loss.losses import  OHEM_loss,hard_negative_mining





class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self,
                 classes,
                 class_agnostic,
                 compact_mode=False):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic


        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        # self.focalloss_handle = FocalLossV4(num_class=21, alpha=0.25, gamma=2.0, balance_index=2)
        # define Large Separable Convolution Layer

        self.rpn = RPN(in_channels=245, f_channels=256)


        self.sam = SAM(256,245)
        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        self.rpn_time = None
        self.pre_roi_time = None
        self.roi_pooling_time = None
        self.subnet_time = None
        self.psroiAlign =  PSROIAlignhandle(1./16, 7,2, 5)
        self.psroiPool =  PSROIPoolhandle(7,7,1./16,7,5)



    def _roi_pool_layer(self, bottom, rois):
        return self.psroiPool.forward(bottom, rois)

    def _roi_align_layer(self, bottom, rois):
        return self.psroiAlign.forward(bottom, rois)

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)


        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        start = time.time()
        basefeat = self.RCNN_base(im_data)

        # feed base feature map tp RPN to obtain rois
        rpn_feat= self.rpn(basefeat)


        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(rpn_feat, im_info, gt_boxes, num_boxes)
        rpn_time = time.time()
        self.rpn_time = rpn_time - start
        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)

        pre_roi_time = time.time()
        self.pre_roi_time = pre_roi_time - rpn_time

        base_feat = self.sam([basefeat,rpn_feat])



        # do roi pooling based on predicted rois
        if cfg.POOLING_MODE == 'align':
            pooled_feat = self._roi_align_layer(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self._roi_pool_layer(base_feat, rois.view(-1, 5))

        roi_pool_time = time.time()
        self.roi_pooling_time = roi_pool_time - pre_roi_time

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0),
                                            int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(
                bbox_pred_view, 1,
                rois_label.view(rois_label.size(0), 1,
                                1).expand(rois_label.size(0), 1, 4))

            bbox_pred = bbox_pred_select.squeeze(1)


        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0


        if self.training:
            # classification loss
            # RCNN_loss_cls = OHEM_loss(cls_score,rois_label)

            loss = -F.log_softmax(cls_score, dim=1)[:, 0]
            mask, num_pos = hard_negative_mining(loss, rois_label)
            confidence = cls_score[mask, :]
            RCNN_loss_cls = F.cross_entropy(confidence, rois_label[mask], reduction='mean')


            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
            RCNN_loss_bbox = RCNN_loss_bbox * 2  # "to balance multi-task training"

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        subnet_time = time.time()
        self.subnet_time = subnet_time - roi_pool_time
        time_measure = [
            self.rpn_time, self.pre_roi_time, self.roi_pooling_time,
            self.subnet_time
        ]

        return time_measure, rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                # not a perfect approximation
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        # normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
