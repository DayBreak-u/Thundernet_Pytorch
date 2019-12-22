
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
from torch.utils.data import  RandomSampler
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data.sampler import Sampler
import random
from roi_data_layer.roidb import combined_roidb

from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir,_merge_a_into_b
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.Snet import snet
from test_net import eval_result
# from model.faster_rcnn.resnet import resnet

from roi_data_layer.roibatchLoader  import Detection
from roi_data_layer.augmentation import SSDAugmentation




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

    # set pretrained model;
    parser.add_argument('--pre',
                        dest='pre',
                        help='pretrained model',
                        default=None,
                        type=str)

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



    # set eval_interval

    parser.add_argument('--eval_interval',
                        dest='eval_interval',
                        help='eval interval',
                        default=5,
                        type=int)

    args = parser.parse_args()
    return args




class BatchSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last = False,size = 320):
        self.size  = size
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.count_change_size   = 0


    def __iter__(self):
        batch = []

        for idx in self.sampler:
            batch.append([idx,self.size])
            if len(batch) == self.batch_size:
                self.count_change_size += 1
                yield batch
                batch = []
                if self.count_change_size  % 10 == 0:
                    self.size = random.choice(cfg.TRAIN.SIZE)
                    # print("change train size to ({},{})".format(self.size,self.size))
        self.count_change_size = 0
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


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
        args.imdb_name = "coco_2017_train"
        args.imdbval_name = "coco_2017_val"
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
    imdb, roidb = combined_roidb(args.imdb_name)
    # imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)

    print('{:d} roidb entries'.format(len(roidb)))

    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)



    # dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
    #                          imdb.num_classes, training=True)

    # dataset = roibatchLoader(roidb , imdb.num_classes, training=True)

    dataset = Detection(roidb,num_classes= imdb.num_classes,
                            transform=SSDAugmentation(cfg.TRAIN.SIZE,
                                                      cfg.PIXEL_MEANS))
    sampler_batch = BatchSampler(RandomSampler(dataset), args.batch_size)


    dataloader = torch.utils.data.DataLoader(dataset,
                                             # batch_size=args.batch_size,
                                             batch_sampler=sampler_batch ,
                                             # shuffle=True,
                                             num_workers=args.num_workers,
                                                )



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
    print(imdb.classes)
    _RCNN = snet(imdb.classes,layer , pretrained_path =args.pre, class_agnostic=args.class_agnostic,)


    _RCNN.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE

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
        args.lr = args.lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    lr = args.lr
    if args.resume:
        load_name = os.path.join(
                output_dir,
                'thundernet_epoch_{}.pth'.format( args.checkepoch,
                                                  ))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)

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

    warm  = True
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        _RCNN.train()
        loss_temp = 0
        start = time.time()

        if epoch in list(map(int,args.lr_decay_step.split(",") )):
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma


        if epoch % args.eval_interval == 0  and epoch>0:
            eval_result(args, logger, epoch-1,output_dir)




        data_iter = iter(dataloader)

        if epoch == 0:
            adjust_learning_rate(optimizer, 0.0001)
            lr *= 0.0001

        for step in range(iters_per_epoch):

            # if step%5==0:
            #     scale = random.choice([320,480,640])
            #     GlobalVar.set_sacle(scale)
                # print(data_iter.dataset.scale)

            if step % 100 == 0 and step > 0 and epoch == 0 and warm:
                adjust_learning_rate(optimizer,10)
                lr *= 10

                if lr  >= args.lr - 0.00001:
                    warm = False

            data = next(data_iter)
            # pdb.set_trace()
            # hm, reg_mask, wh
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
            rois_label = _RCNN(im_data, im_info, gt_boxes, num_boxes,
                               # hm,reg_mask,wh,offset,ind
                               )

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

                print("[epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e, rpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f size:(%3d,%3d)" \
                    % (epoch, step, iters_per_epoch, loss_temp, lr, loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box,sampler_batch.size,sampler_batch.size))
                # scale = random.choice([256, 320, 480])
                # cfg.TRAIN.SCALES = [scale]
                # print("change SCALE:{}".format(scale))
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
                    'thundernet_epoch_{}.pth'.format( epoch,
                                                  ))
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'model': _RCNN.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'pooling_mode': cfg.POOLING_MODE,
                    'class_agnostic': args.class_agnostic,
                }, save_name)
        else:

            save_name = os.path.join(
                    output_dir,
                    'thundernet_epoch_{}.pth'.format(epoch,
                                                 ))
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'model': _RCNN.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'pooling_mode': cfg.POOLING_MODE,
                    'class_agnostic': args.class_agnostic,
                }, save_name)
        print('save model: {}'.format(save_name))

        end = time.time()
        print("Average time per iter: {}".format(end - start))
