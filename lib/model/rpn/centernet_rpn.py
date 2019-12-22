from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.config import cfg
from model.loss.losses import FocalLoss,RegL1Loss
from torch.autograd import Variable
from  model.utils.cente_decode import ctdet_decode


class _ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, feat_stride):
        super(_ProposalLayer, self).__init__()

        self._feat_stride = feat_stride


        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        # top[0].reshape(1, 5)
        #
        # # scores blob: holds scores for R regions of interest
        # if len(top) > 1:
        #     top[1].reshape(1, 1, 1, 1)

    def forward(self, input):

        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)


        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs
        scores = input[0]
        wh_deltas = input[1]
        offset_deltas = input[2]
        im_info = input[3]
        cfg_key = input[4]



        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N

        detections = ctdet_decode(scores,wh_deltas,offset_deltas,K=post_nms_topN)




        detections[:, :, :4] *=  self._feat_stride
        batch_size = scores.size(0)



        output = scores.new(batch_size, post_nms_topN, 5).zero_()

        for i in range(batch_size):
            # # 3. remove predicted boxes with either height or width < threshold
            # # (NOTE: convert min_size to input image scale stored in im_info[2])

            output[i,:,0] = i

            output[i,:,1:] = detections[i,:,:4]

        return output

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
        hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
        keep = ((ws >= min_size.view(-1,1).expand_as(ws)) & (hs >= min_size.view(-1,1).expand_as(hs)))
        return keep




class _RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din):
        super(_RPN, self).__init__()

        self.din = din  # get depth of input feature map, e.g., 512

        self.feat_stride = cfg.FEAT_STRIDE


        self.RPN_hm_score = nn.Conv2d(self.din, 1 , 1, 1, 0)
        self.PRN_wh_score = nn.Conv2d(self.din, 2 , 1, 1, 0)
        self.PRN_offset_score = nn.Conv2d(self.din, 2 , 1, 1, 0)
        self.RPN_proposal = _ProposalLayer(self.feat_stride)

        self.crit = FocalLoss()
        # self.crit =torch.nn.MSELoss()
        self.crit_offset = RegL1Loss()
        self.crit_wh = RegL1Loss()

        self.rpn_loss_hm = 0
        self.rpn_loss_wh = 0
        self.rpn_loss_offset = 0


    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(input_shape[0], int(d),
                   int(float(input_shape[1] * input_shape[2]) / float(d)),
                   input_shape[3])
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes,hm,reg_mask,wh,offset,ind):

        batch_size = base_feat.size(0)

        rpn_hm_score = self.RPN_hm_score(base_feat)
        rpn_cls_prob = F.sigmoid(rpn_hm_score)
        rpn_wh_pred = self.PRN_wh_score(base_feat)
        rpn_offset_pred = self.PRN_offset_score(base_feat)

        cfg_key = 'TRAIN' if self.training else 'TEST'



        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None

            hm_loss =  self.crit(rpn_cls_prob, hm)

            offset_loss = self.crit_offset(rpn_offset_pred, reg_mask,
                          ind, offset)

            wh_loss = self.crit_wh(rpn_wh_pred, reg_mask,
                                       ind, wh)

            self.rpn_loss_cls = hm_loss + offset_loss
            self.rpn_loss_box =  wh_loss

        rois = self.RPN_proposal(
            (rpn_cls_prob, rpn_wh_pred, rpn_offset_pred, im_info, cfg_key))
        return rois, self.rpn_loss_cls, self.rpn_loss_box
