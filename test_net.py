# --------------------------------------------------------
# Pytorch Multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Lichao Wang, based on code from Ross Girshick, Jiasen Lu, Jianwei Yang
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as trans
# import torch._utils as utils
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from torchvision.ops import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.Snet import snet
from PIL import  Image
# import pdb
from utils import  color_list

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3
"""
# Support older version models for PyTorch
try:
  utils._rebuild_tensor_v2
  _v2_flag = False
except AttributeError:
  def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
    tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
    tensor.requires_grad = requires_grad
    tensor._backward_hooks = backward_hooks
    return tensor
  utils._rebuild_tensor_v2 = _rebuild_tensor_v2
  _v2_flag = True
"""


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(
        description='Train a Faster R-CNN network')
    parser.add_argument('--dataset',
                        dest='dataset',
                        help='training dataset',
                        default='pascal_voc',
                        type=str)
    parser.add_argument('--cfg',
                        dest='cfg_file',
                        help='optional config file',
                        default='cfgs/res101_ls.yml',
                        type=str)
    parser.add_argument('--net',
                        dest='net',
                        help='vgg16, res50, res101, res152, xception',
                        default='res101',
                        type=str)
    parser.add_argument('--set',
                        dest='set_cfgs',
                        help='set config keys',
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir',
                        dest='load_dir',
                        help='directory to load models',
                        default="models",
                        type=str)
    parser.add_argument('--cuda',
                        dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls',
                        dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs',
                        dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag',
                        dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument(
        '--parallel_type',
        dest='parallel_type',
        help=
        'which part of model to parallel, 0: all, 1: model before roi pooling',
        default=0,
        type=int)

    parser.add_argument('--checkepoch',
                        dest='checkepoch',
                        help='checkepoch to load network',
                        default=1,
                        type=int)

    parser.add_argument('--bs',
                        dest='batch_size',
                        help='batch_size',
                        default=1,
                        type=int)
    parser.add_argument('--vis',
                        dest='vis',
                        help='visualization mode',
                        action='store_true')

    # lighthead mode
    parser.add_argument('--lighthead',
                        dest='lighthead',
                        help='whether to use light-head R-CNN',
                        action='store_true')

    args = parser.parse_args()
    return args

def eval_result(args,logger,epoch,output_dir):
    if torch.cuda.is_available() and not args.cuda:
        print(
            "WARNING: You have a CUDA device, so you should probably run with --cuda"
        )

    args.batch_size = 1
    imdb, roidb, ratio_list, ratio_index = combined_roidb(
        args.imdbval_name, False)

    imdb.competition_mode(on=True)


    load_name = os.path.join(
            output_dir,
            'thundernet_epoch_{}.pth'.format( epoch,
                                                  ))


    layer = int(args.net.split("_")[1])
    _RCNN = snet(imdb.classes, layer, pretrained_path=None, class_agnostic=args.class_agnostic)


    _RCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    if args.cuda:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name,
                                map_location=lambda storage, loc: storage
                                )  # Load all tensors onto the CPU
    _RCNN.load_state_dict(checkpoint['model'])




    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable (PyTorch 0.4.0+)
    with torch.no_grad():
        im_data = Variable(im_data)
        im_info = Variable(im_info)
        num_boxes = Variable(num_boxes)
        gt_boxes = Variable(gt_boxes)

    if args.cuda:
        cfg.CUDA = True

    if args.cuda:
        _RCNN.cuda()

    start = time.time()
    max_per_image = 100

    vis = True

    if vis:
        thresh = 0.05
    else:
        thresh = 0.0

    save_name = 'thundernet'
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, save_name)
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                             imdb.num_classes, training=False, normalize=False)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=0,
                                             pin_memory=True)

    data_iter = iter(dataloader)

    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(output_dir, 'detections.pkl')

    _RCNN.eval()


    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))

    for i in range(num_images):

        data = next(data_iter)

        with torch.no_grad():
            im_data.resize_(data[0].size()).copy_(data[0])
            im_info.resize_(data[1].size()).copy_(data[1])
            gt_boxes.resize_(data[2].size()).copy_(data[2])
            num_boxes.resize_(data[3].size()).copy_(data[3])

        det_tic = time.time()
        with torch.no_grad():
            time_measure, \
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = _RCNN(im_data, im_info, gt_boxes, num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(args.batch_size, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(args.batch_size, -1, 4 * len(imdb.classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= data[1][0][2].item()

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        if vis:
            im = cv2.imread(imdb.image_path_at(i))
            im2show = np.copy(im)
        for j in xrange(1, imdb.num_classes):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order],
                           cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if vis:
                    vis_detections(im2show, imdb.classes[j], color_list[j].tolist()  ,
                                             cls_dets.cpu().numpy(), 0.6)
                all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][i] = empty_array

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack(
                [all_boxes[j][i][:, -1] for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        sys.stdout.write(
            'im_detect: {:d}/{:d}\tDetect: {:.3f}s (RPN: {:.3f}s, Pre-RoI: {:.3f}s, RoI: {:.3f}s, Subnet: {:.3f}s)\tNMS: {:.3f}s\r' \
            .format(i + 1, num_images, detect_time, time_measure[0], time_measure[1], time_measure[2],
                    time_measure[3], nms_time))
        sys.stdout.flush()

        if vis and i%200 == 0 and args.use_tfboard:
            im2show = im2show[:,:,::-1]
            logger.add_image('pred_image_{}'.format(i), trans.ToTensor()(Image.fromarray(im2show.astype('uint8'))), global_step= i)


            # cv2.imwrite('result.png', im2show)
            # pdb.set_trace()
            # cv2.imshow('test', im2show)
            # cv2.waitKey(0)

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    ap_50 = imdb.evaluate_detections(all_boxes, output_dir)
    logger.add_scalar("map_50" ,
                     ap_50, global_step = epoch)

    end = time.time()
    print("test time: %0.4fs" % (end - start))




