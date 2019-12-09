import torch.nn as nn
import torch

import torch.nn.functional as F
import functools



CEM_FILTER = 245
anchor_number = 25

def conv_bn(inp, oup, kernel_size , stride , pad):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )



class CEM(nn.Module):
    """Context Enhancement Module"""

    def __init__(self, in_channels1, in_channels2 ,in_channels3 , kernel_size=1, stride=1):
        super(CEM, self).__init__()

        downsample_size = CEM_FILTER

        self.conv4 = nn.Conv2d(in_channels1, downsample_size, kernel_size, bias=True)

        self.conv5 = nn.Conv2d(in_channels2, downsample_size, kernel_size, bias=True)

        self.convlast= nn.Conv2d(in_channels3, downsample_size, kernel_size, bias=True)


        # self.conv7 = nn.Conv2d(downsample_size, CEM_FILTER, kernel_size = 3, stride = 1, padding= 2 , bias=True)
        # self.bn7 = nn.BatchNorm2d(CEM_FILTER)
        self.relu7 = nn.ReLU(inplace=True)

    def forward(self,inputs):

        C4_lat = self.conv4(inputs[0])

        C5_lat = self.conv5(inputs[1])

        C5_lat = F.interpolate(C5_lat, size=[C4_lat.size(2), C4_lat.size(3)], mode="nearest")

        C6_lat = self.convlast(inputs[2])


        out = C4_lat + C5_lat + C6_lat

        # out = self.conv7(out)
        # out = self.bn7(out)
        out = self.relu7(out)

        # return C4_lat,out
        return out


#
# class FPN(nn.Module):
#     """Context Enhancement Module"""
#
#     def __init__(self, in_channels1, in_channels2 ,in_channels3 ,backone, kernel_size=1, stride=1):
#         super(FPN, self).__init__()
#         self.deconv_with_bias = False
#         self.backone  = backone
#         self.conv4 = nn.Conv2d(in_channels1, CEM_FILTER, kernel_size, bias=True)
#         self.conv5 = nn.Conv2d(in_channels2, CEM_FILTER, kernel_size, bias=True)
#         self.conv6 = nn.Conv2d(in_channels3, CEM_FILTER, kernel_size, bias=True)
#
#         self.deconv6 = nn.Sequential(nn.ConvTranspose2d(
#             in_channels=CEM_FILTER,
#             out_channels=CEM_FILTER,
#             kernel_size=2,
#             stride=2,
#             padding=0,
#             output_padding=0,
#             bias=self.deconv_with_bias),
#             nn.BatchNorm2d(CEM_FILTER),
#             nn.ReLU(inplace=True)
#         )
#
#         self.deconv5 = nn.Sequential(nn.ConvTranspose2d(
#             in_channels=CEM_FILTER,
#             out_channels=CEM_FILTER,
#             kernel_size=2,
#             stride=2,
#             padding=0,
#             output_padding=0,
#             bias=self.deconv_with_bias),
#             nn.BatchNorm2d(CEM_FILTER),
#             nn.ReLU(inplace=True)
#         )
#
#
#
#     def forward(self,x):
#         # in keras NHWC
#         # in torch NCHW
#
#         inputs = self.backone(x)
#         c6 = self.conv6(inputs[2])
#         c6 = self.deconv6(c6)
#         c5 = self.conv5(inputs[1])
#
#         c5 = c5 + c6
#
#         c5 = self.deconv5(c5)
#
#         c4 = self.conv4(inputs[0])
#         c4 = c4 + c5
#
#
#
#         return c4


class RPN(nn.Module):
    """region proposal network"""

    def __init__(self, in_channels=245, f_channels=256):
        super(RPN, self).__init__()
        self.num_anchors = anchor_number

        self.dw5_5 =  nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1, padding=2, groups=in_channels)
        self.bn0 = nn.BatchNorm2d(in_channels)
        self.relu0 = nn.ReLU(inplace=True)
        self.con1x1 = nn.Conv2d(in_channels, f_channels, kernel_size=1)
        self.relu1 = nn.ReLU(inplace=True)



    def forward(self, x):

        x = self.dw5_5(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.con1x1(x)
        x = self.relu1(x)

        return x


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
        out = cem * sam
        return out



class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride==1:
            x_proj, x = self.channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride==2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    def channel_shuffle(self, x):

        g = 2
        # n, c, h, w = x.size()
        # x = x.view(n, g, c // g, h, w).permute(
        #     0, 2, 1, 3, 4).contiguous().view(n, c, h, w)

        x = x.reshape(x.shape[0], g, x.shape[1] // g, x.shape[2], x.shape[3])
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])
        x_proj = x[:, :(x.shape[1] // 2), :, :]
        x = x[:, (x.shape[1] // 2):, :, :]
        return x_proj, x