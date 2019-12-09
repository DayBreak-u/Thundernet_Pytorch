# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import errno
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.Snet import snet
from test_net import eval_result

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
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
                        help='snet_49, snet_146',
                        default='res101',
                        type=str)
    parser.add_argument('--set',
                        dest='set_cfgs',
                        help='set config keys',
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--start_epoch',
                        dest='start_epoch',
                        help='starting epoch',
                        default=0,
                        type=int)
    parser.add_argument('--epochs',
                        dest='max_epochs',
                        help='number of epochs to train',
                        default=20,
                        type=int)
    parser.add_argument('--disp_interval',
                        dest='disp_interval',
                        help='number of iterations to display',
                        default=100,
                        type=int)
    parser.add_argument('--checkpoint_interval',
                        dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000,
                        type=int)

    parser.add_argument('--save_dir',
                        dest='save_dir',
                        help='directory to save models',
                        default="models",
                        type=str)
    parser.add_argument('--nw',
                        dest='num_workers',
                        help='number of worker to load data',
                        default=0,
                        type=int)
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
    parser.add_argument('--bs',
                        dest='batch_size',
                        help='batch_size',
                        default=1,
                        type=int)
    parser.add_argument('--cag',
                        dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')

    # config optimization
    parser.add_argument('--o',
                        dest='optimizer',
                        help='training optimizer',
                        default="sgd",
                        type=str)
    parser.add_argument('--lr',
                        dest='lr',
                        help='starting learning rate',
                        default=0.001,
                        type=float)
    parser.add_argument('--lr_decay_step',
                        dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default="100,150,200",
                        type=str)
    parser.add_argument('--lr_decay_gamma',
                        dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1,
                        type=float)

    # set training session
    parser.add_argument('--s',
                        dest='session',
                        help='training session',
                        default=1,
                        type=int)

    # resume trained model
    parser.add_argument('--r',
                        dest='resume',
                        help='resume checkpoint or not',
                        default=False,
                        type=bool)

    parser.add_argument('--checkepoch',
                        dest='checkepoch',
                        help='checkepoch to load model',
                        default=1,
                        type=int)

    # log and diaplay
    parser.add_argument('--logdir',
                        dest='logdir',
                        help='logdir',
                        default="logs",
                        type=str)

    parser.add_argument('--use_tfboard',
                        dest='use_tfboard',
                        help='whether use tensorflow tensorboard',
                        default=False,
                        type=bool)



    # set eval session

    parser.add_argument('--eval_interval',
                        dest='eval_interval',
                        help='eval interval',
                        default=5,
                        type=int)

    args = parser.parse_args()
    return args


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch * batch_size,
                                         train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1,
                                                           1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch,
                                        self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),
                                           0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data


if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)




    if args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        # args.set_cfgs = [
        #     'ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]',
        #     'MAX_NUM_GT_BOXES', '20'
        # ]
        args.set_cfgs = [
            'ANCHOR_SCALES', '[2, 4 , 8, 16, 32]', 'ANCHOR_RATIOS', '[1.0/2 , 3.0/4 , 1 , 4.0/3 , 2 ]',
            'MAX_NUM_GT_BOXES', '20'
        ]
    elif args.dataset == "coco":
        args.imdb_name = "tian2017"
        args.imdbval_name = "val2017"
        args.set_cfgs = [
            'ANCHOR_SCALES', '[2, 4 , 8, 16, 32]', 'ANCHOR_RATIOS', '[1.0/2 , 3.0/4 , 1 , 4.0/3 , 2 ]',
            'MAX_NUM_GT_BOXES', '50'
        ]


    args.cfg_file = "cfgs/{}_ls.yml".format(
        args.net) if args.large_scale else "cfgs/{}.yml".format(args.net.split("_")[0])

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print(
            "WARNING: You have a CUDA device, so you should probably run with --cuda"
        )

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)

    print('{:d} roidb entries'.format(len(roidb)))

    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sampler_batch = sampler(train_size, args.batch_size)

    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                             imdb.num_classes, training=True)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             sampler=sampler_batch,
                                             num_workers=args.num_workers)



    val_data_iter = iter(dataloader)

    # initilize the tensor holder here.
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

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda:
        cfg.CUDA = True



    # initilize the network here.

    layer = int(args.net.split("_")[1])
    _RCNN = snet(imdb.classes,layer , pretrained=False, class_agnostic=args.class_agnostic,)


    _RCNN.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr
    # tr_momentum = cfg.TRAIN.MOMENTUM
    # tr_momentum = args.momentum

    params = []
    for key, value in dict(_RCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{
                    'params': [value],
                    'lr': lr,
                    'weight_decay': cfg.TRAIN.WEIGHT_DECAY
                }]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.resume:
        load_name = os.path.join(
                output_dir,
                'thundernet_epoch_{}.pth'.format( args.checkepoch,
                                                  ))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        _RCNN.load_state_dict(checkpoint['model'],strict=False)
        # optimizer.load_state_dict(checkpoint['optimizer'])

        # lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))

    # checkpoint = torch.load("snet_49/snet_49/pascal_voc_0712/lighthead_rcnn_1_160_1033.pth")
    # _RCNN.load_state_dict(checkpoint['model'], strict=False)

    if args.mGPUs:
        _RCNN = nn.DataParallel(_RCNN)

    if args.cuda:
        _RCNN.cuda()

    iters_per_epoch = int(train_size / args.batch_size)

    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        # from torch.utils.tensorboard import SummaryWriter
        logger = SummaryWriter(args.logdir)

    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode

        loss_temp = 0
        start = time.time()

        if epoch in list(map(int,args.lr_decay_step.split(",") )):
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma


        if epoch % args.eval_interval == 0  and epoch>0:
            eval_result(args, logger, epoch-1,output_dir)

        _RCNN.train()


        data_iter = iter(dataloader)
        for step in range(iters_per_epoch):
            data = next(data_iter)

            with torch.no_grad():
                im_data.resize_(data[0].size()).copy_(data[0])
                im_info.resize_(data[1].size()).copy_(data[1])
                gt_boxes.resize_(data[2].size()).copy_(data[2])
                num_boxes.resize_(data[3].size()).copy_(data[3])


            _RCNN.zero_grad()
            time_measure, \
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = _RCNN(im_data, im_info, gt_boxes, num_boxes)

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                   + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            loss_temp += loss.data.item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            #if args.net == "vgg16":
            #    clip_gradient(_RCNN, 10.)
            optimizer.step()

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= args.disp_interval

                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                print("[epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e, rpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                    % (epoch, step, iters_per_epoch, loss_temp, lr, loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
                # print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                #       % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
                # print("\t\tfg/bg=(%d/%d), time cost: %.3f sec" %
                #       (fg_cnt, bg_cnt, end - start))
                # print("\t\tTime Details: RPN: %.3f, Pre-RoI: %.3f, RoI: %.3f, Subnet: %.3f" \
                #       % (time_measure[0], time_measure[1], time_measure[2], time_measure[3]))
                # print("\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                #       % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))

                if args.use_tfboard:
                    info = {
                        'Total Loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_rcnn_box': loss_rcnn_box,
                        'Learning Rate': lr,
                        'Time Cost': end - start
                    }
                    for tag, value in info.items():
                        logger.add_scalar(
                            tag, value, step + ((epoch - 1) * iters_per_epoch))

                loss_temp = 0
                start = time.time()

        if args.mGPUs:

            save_name = os.path.join(
                    output_dir,
                    'thundernet_epoch_{}.pth'.format( args.checkepoch,
                                                  ))
            save_checkpoint(
                {
                    'session': args.session,
                    'epoch': epoch + 1,
                    'model': _RCNN.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'pooling_mode': cfg.POOLING_MODE,
                    'class_agnostic': args.class_agnostic,
                }, save_name)
        else:

            save_name = os.path.join(
                    output_dir,
                    'thundernet_epoch_{}.pth'.format(args.checkepoch,
                                                 ))
            save_checkpoint(
                {
                    'session': args.session,
                    'epoch': epoch + 1,
                    'model': _RCNN.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'pooling_mode': cfg.POOLING_MODE,
                    'class_agnostic': args.class_agnostic,
                }, save_name)
        print('save model: {}'.format(save_name))

        end = time.time()
        print("Average time per iter: {}".format(end - start))
