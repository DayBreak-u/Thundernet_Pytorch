import torch.nn as nn
import torch
from config import Configs
import torch.nn.functional as F
import functools


CEM_FILTER = Configs.get("CEM_FILTER")
anchor_sizes = Configs.get("anchor_sizes")
aspect_ratios = Configs.get("aspect_ratios")
anchor_number = Configs.get("anchor_number")

def conv_bn(inp, oup, kernel_size , stride , pad):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )



def conv_dw(inp, oup, kernel_size , stride , pad):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size, stride, pad, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )




class CEM(nn.Module):
    """Context Enhancement Module"""

    def __init__(self, in_channels1, in_channels2 ,in_channels3 ,backone, kernel_size=1, stride=1):
        super(CEM, self).__init__()
        self.backone  = backone
        self.conv4 = nn.Conv2d(in_channels1, CEM_FILTER, kernel_size, bias=True)
        self.conv5 = nn.Conv2d(in_channels2, CEM_FILTER, kernel_size, bias=True)
        self.convlast = nn.Conv2d(in_channels3, CEM_FILTER, kernel_size, bias=True)
        self.unsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self,x):
        # in keras NHWC
        # in torch NCHW

        inputs = self.backone(x)
        C4_lat = self.conv4(inputs[0])
        C5_lat = self.conv5(inputs[1])
        C5_lat = self.unsample(C5_lat)
        Cglb_lat = self.convlast(inputs[2])

        return C4_lat + C5_lat + Cglb_lat

#
# class RPN(nn.Module):
#     """region proposal network"""
#
#     def __init__(self, in_channels=245, f_channels=256):
#         super(RPN, self).__init__()
#         self.num_anchors = anchor_number
#
#         self.dw5_5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1, padding=2, groups=in_channels)
#         self.bn0 = nn.BatchNorm2d(in_channels)
#         self.con1x1 = nn.Conv2d(in_channels, f_channels, kernel_size=1)
#         self.bn1 = nn.BatchNorm2d(f_channels)
#
#         self.conv1 = nn.Conv2d(f_channels, in_channels, kernel_size=1)
#         self.bn2 = nn.BatchNorm2d(in_channels)
#
#         self.loc_conv = nn.Conv2d(in_channels, self.num_anchors, kernel_size=1, stride=1,
#                                   padding=0
#                                   )
#         self.rpn_cls_pred = nn.Conv2d(in_channels, 4 * self.num_anchors, kernel_size=1,
#                                       stride=1, padding=0
#                                       )
#
#         for l in self.children():
#             torch.nn.init.normal_(l.weight, std=0.01)
#             torch.nn.init.constant_(l.bias, 0)
#
#     def forward(self, x):
#         logits = []
#         bbox_reg = []
#
#         temp = self.dw5_5(x)  # SAM
#         temp = self.bn0(temp)  # SAM
#         temp = F.relu(temp)  # SAM
#         temp = self.con1x1(temp)
#         temp = self.bn1(temp)
#         temp = F.relu(temp)
#
#         temp = self.conv1(temp)
#         temp = self.bn2(temp)
#         temp = F.sigmoid(temp)
#         temp = x.mul(temp)
#
#         logits.append(self.loc_conv(temp))
#         bbox_reg.append(self.rpn_cls_pred(temp))
#         return logits, bbox_reg


class RPN(nn.Module):
    """region proposal network"""

    def __init__(self, in_channels=245, f_channels=256):
        super(RPN, self).__init__()
        self.num_anchors = anchor_number

        self.dw5_5 =  nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1, padding=2, groups=in_channels)
        self.bn0 = nn.BatchNorm2d(in_channels)
        self.con1x1 = nn.Conv2d(in_channels, f_channels, kernel_size=1)


        self.rpn_cls_pred = nn.Conv2d(f_channels,  self.num_anchors, kernel_size=1, stride=1,
                               padding=0
                               )
        self.loc_conv = nn.Conv2d(f_channels, 4 * self.num_anchors, kernel_size=1,
                                   stride=1, padding=0
                                   )



    def forward(self, x):
        logits = []
        bbox_reg = []
        x = self.dw5_5(x)
        x = self.bn0(x)
        x = self.con1x1(x)

        logits.append(self.rpn_cls_pred(x))
        bbox_reg.append(self.loc_conv(x))



        return logits, bbox_reg , x


class SAM(torch.nn.Module):
    def __init__(self,  f_channels ,CEM_FILTER ):
        super(SAM, self).__init__()

        self.conv1 = nn.Conv2d(f_channels, CEM_FILTER, kernel_size=1)
        self.bn = nn.BatchNorm2d(CEM_FILTER)

    def forward(self, input):

        cem = input[0]
        rpn = input[1]

        sam = self.conv1(rpn)
        sam = self.bn(sam)
        sam = F.sigmoid(sam)
        out = cem.mul(sam)
        return out





class BasicBlock(nn.Module):
    def __init__(self, in_channels, shuffle_groups=2):
        super().__init__()
        channels = in_channels // 2
        self.channels = channels
        self.conv1 = conv_bn(
            channels, channels, kernel_size=1,stride=1, pad= 0
        )
        self.conv2 = conv_dw(
            channels, channels, kernel_size=5,  stride=1, pad= 2
        )
        self.conv3 = conv_bn(
            channels, channels, kernel_size=1,stride=1, pad= 0
        )

        self.shuffle = ShuffleBlock(shuffle_groups)

    def forward(self, x):
        x = x.contiguous()
        c = x.size(1) // 2

        x1 = x[:, :c, :, :]
        x2 = x[:, c:, :, :]
        x2 = self.conv1(x2)
        x2 = self.conv2(x2)
        x2 = self.conv3(x2)
        # if self.with_se:
        #     x2 = self.se(x2)
        # print(x1.shape)
        # print(x2.shape)

        x = torch.cat((x1, x2), dim=1)
        x = self.shuffle(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, shuffle_groups=2, **kwargs):
        super().__init__()
        channels = out_channels // 2
        self.conv11 = conv_dw(
            in_channels, in_channels, kernel_size=5, stride=2, pad= 2
        )
        self.conv12 = conv_bn(
            in_channels, channels, kernel_size=1, stride=1, pad= 0
        )
        self.conv21 = conv_bn(
            in_channels, channels, kernel_size=1, stride=1, pad= 0
        )
        self.conv22 = conv_dw(
            channels, channels, kernel_size=5, stride=2, pad= 2
        )
        self.conv23 = conv_bn(
            channels, channels, kernel_size=1,stride=1, pad= 0
        )
        self.shuffle = ShuffleBlock(shuffle_groups)

    def forward(self, x):
        x1 = self.conv11(x)

        x1 = self.conv12(x1)

        x2 = self.conv21(x)
        x2 = self.conv22(x2)
        x2 = self.conv23(x2)

        x = torch.cat((x1, x2), dim=1)
        x = self.shuffle(x)
        return x


def channel_shuffle(x, g):
    n, c, h, w = x.size()
    x = x.view(n, g, c // g, h, w).permute(
        0, 2, 1, 3, 4).contiguous().view(n, c, h, w)
    return x


'''
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x
'''


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        return channel_shuffle(x, g=self.groups)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        pw_conv11 = functools.partial(nn.Conv2d, kernel_size=1, stride=1, padding=0, bias=False)
        dw_conv33 = functools.partial(self.depthwise_conv,
                                      kernel_size=3, stride=self.stride, padding=1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                dw_conv33(inp, inp),
                nn.BatchNorm2d(inp),
                pw_conv11(inp, branch_features),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            pw_conv11(inp if (self.stride > 1) else branch_features, branch_features),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            dw_conv33(branch_features, branch_features),
            nn.BatchNorm2d(branch_features),
            pw_conv11(branch_features, branch_features),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out
