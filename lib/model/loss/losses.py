import torch
from torch.nn import functional as F
import pdb
import  math

def _smooth_l1_loss(bbox_pred,
                    bbox_targets,
                    bbox_inside_weights,
                    bbox_outside_weights,
                    sigma=1.0,
                    dim=[1],
                    reduce = True):

    sigma_2 = sigma**2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    # print("faster rcnn:")
    # print(loss_box.shape)
    for i in sorted(dim, reverse=True):
        loss_box = loss_box.sum(i)
    if reduce:
        loss_box = loss_box.mean()
    # print(loss_box)
    return loss_box
#
def OHEM_loss(roi_scores,
              gt_roi_labels,
              n_ohem_sample= 256 ):
    n_sample = roi_scores.shape[0]

    roi_cls_loss = F.cross_entropy(roi_scores, gt_roi_labels, reduce=False)


    n_ohem_sample = min(n_ohem_sample, n_sample)

    total_roi_cls_loss = roi_cls_loss.detach()
    roi_cls_loc_loss = total_roi_cls_loss
    _, indices = roi_cls_loc_loss.sort(descending=True)
    indices = indices[:n_ohem_sample]
    # indices = cuda.to_gpu(indices)

    roi_cls_loss = torch.sum(roi_cls_loss[indices]) / n_ohem_sample

    return roi_cls_loss

def hard_negative_mining(loss, labels, neg_pos_ratio=3):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.
    Args:
        loss (N, num_anchors): the loss for each example.
        labels (N, num_anchors): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """


    pos_mask = labels > 0

    num_pos = pos_mask.long().sum(dim=0, keepdim=True)

    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort( descending=True)
    _, orders = indexes.sort()
    neg_mask = orders < num_neg
    return pos_mask | neg_mask , num_pos