import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                         nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))


def conv_1x1_bn(inp, oup):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                         nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio,
                      inp * expand_ratio,
                      3,
                      stride,
                      1,
                      groups=inp * expand_ratio,
                      bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes,
                               planes,
                               kernel_size=1,
                               stride=stride,
                               bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,  # change
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class LargeSeparableConv2d(nn.Module):
    def __init__(self, c_in, kernel_size=15, bias=False, bn=False,
                 setting='L'):
        super(LargeSeparableConv2d, self).__init__()

        dim_out = 10 * 7 * 7
        c_mid = 64 if setting == 'S' else 256

        self.din = c_in
        self.c_mid = c_mid
        self.c_out = dim_out
        self.k_width = (kernel_size, 1)
        self.k_height = (1, kernel_size)
        self.pad = 0
        self.bias = bias
        self.bn = bn

        self.block1_1 = nn.Conv2d(self.din,
                                  self.c_mid,
                                  self.k_width,
                                  1,
                                  padding=self.pad,
                                  bias=self.bias)
        self.bn1_1 = nn.BatchNorm2d(self.c_mid)
        self.block1_2 = nn.Conv2d(self.c_mid,
                                  self.c_out,
                                  self.k_height,
                                  1,
                                  padding=self.pad,
                                  bias=self.bias)
        self.bn1_2 = nn.BatchNorm2d(self.c_out)

        self.block2_1 = nn.Conv2d(self.din,
                                  self.c_mid,
                                  self.k_height,
                                  1,
                                  padding=self.pad,
                                  bias=self.bias)
        self.bn2_1 = nn.BatchNorm2d(self.c_mid)
        self.block2_2 = nn.Conv2d(self.c_mid,
                                  self.c_out,
                                  self.k_width,
                                  1,
                                  padding=self.pad,
                                  bias=self.bias)
        self.bn2_2 = nn.BatchNorm2d(self.c_out)

    def forward(self, x):
        x1 = self.block1_1(x)
        x1 = self.bn1_1(x1) if self.bn else x1
        x1 = self.block1_2(x1)
        x1 = self.bn1_2(x1) if self.bn else x1

        x2 = self.block2_1(x)
        x2 = self.bn2_1(x2) if self.bn else x2
        x2 = self.block2_2(x2)
        x2 = self.bn2_2(x2) if self.bn else x2

        return x1 + x2


class SeparableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels,
                               in_channels,
                               kernel_size,
                               stride,
                               padding,
                               dilation,
                               groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels,
                                   out_channels,
                                   1,
                                   1,
                                   0,
                                   1,
                                   1,
                                   bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class _Block(nn.Module):
    def __init__(self,
                 in_filters,
                 out_filters,
                 reps,
                 strides=1,
                 start_with_relu=True,
                 grow_first=True):
        super(_Block, self).__init__()

        # Do not use pre-activation design (no identity mappings!)
        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters

        if grow_first:
            rep.append(self.relu)
            rep.append(
                SeparableConv2d(in_filters,
                                out_filters,
                                3,
                                stride=strides,
                                padding=1,
                                bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(
                SeparableConv2d(filters,
                                filters,
                                3,
                                stride=1,
                                padding=1,
                                bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(
                SeparableConv2d(in_filters,
                                out_filters,
                                3,
                                stride=strides,
                                padding=1,
                                bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        # Scaling already applied by stride
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        return x