import time
import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable

def conv_bn(inp, oup, stride = 1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_bn1x1(inp, oup, stride = 1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )



def conv_dw(inp, oup, k , stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class FPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1x1(in_channels_list[0], out_channels, stride = 1)
        self.output2 = conv_bn1x1(in_channels_list[1], out_channels, stride = 1)
        self.output3 = conv_bn1x1(in_channels_list[2], out_channels, stride = 1)

        self.merge1 = conv_bn(out_channels, out_channels)
        self.merge2 = conv_bn(out_channels, out_channels)

    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values())

        output1_ = self.output1(input[0])
        output2_ = self.output2(input[1])
        output3_ = self.output3(input[2])

        up3 = F.interpolate(output3_, size=[output2_.size(2), output2_.size(3)], mode="nearest")
        output2 = output2_ + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1_.size(2), output1_.size(3)], mode="nearest")
        output1 = output1_ + up2
        output1 = self.merge1(output1)

        # out = [output1, output2]
        out = [output1, output2, output3_]
        return out


#
# class MobileNetV1(nn.Module):
#     def __init__(self):
#         super(MobileNetV1, self).__init__()
#         self.stage1 = nn.Sequential(
#             conv_bn(3, 8, 2),    # 3
#             conv_dw(8, 16, 1),   # 7
#             conv_dw(16, 32, 2),  # 11
#             conv_dw(32, 32, 1),  # 19
#             conv_dw(32, 64, 2),  # 27
#             conv_dw(64, 64, 1),  # 43
#         )
#         self.stage2 = nn.Sequential(
#             conv_dw(64, 128, 2),  # 43 + 16 = 59
#             conv_dw(128, 128, 1), # 59 + 32 = 91
#             conv_dw(128, 128, 1), # 91 + 32 = 123
#             conv_dw(128, 128, 1), # 123 + 32 = 155
#             conv_dw(128, 128, 1), # 155 + 32 = 187
#             conv_dw(128, 128, 1), # 187 + 32 = 219
#         )
#         self.stage3 = nn.Sequential(
#             conv_dw(128, 256, 2), # 219 +3 2 = 241
#             conv_dw(256, 256, 1), # 241 + 64 = 301
#         )
#         self.avg = nn.AdaptiveAvgPool2d((1,1))
#         self.fc = nn.Linear(256, 1000)
#
#     def forward(self, x):
#         x = self.stage1(x)
#         x = self.stage2(x)
#         x = self.stage3(x)
#         x = self.avg(x)
#         # x = self.model(x)
#         x = x.view(-1, 256)
#         x = self.fc(x)
#         return x


class MobileNetV1(nn.Module):
    def __init__(self) -> object:
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 16, 2),    # 3
            # conv_dw(16, 16, 1),   # 7
            conv_bn1x1(16, 16, 1),   # 7
            conv_dw(16, 32, 2),  # 11
            conv_dw(32, 32, 1),  # 19
            conv_dw(32, 64, 2),  # 27
            conv_dw(64, 64, 1),  # 43
            conv_dw(64, 64, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1), # 59 + 32 = 91
            conv_dw(128, 128, 1), # 91 + 32 = 123
            conv_dw(128, 128, 1), # 123 + 32 = 155
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 128, 2), # 219 +3 2 = 241
            conv_dw(128, 128, 1), # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.conv1x1 = conv_bn1X1(128, 245, 1)

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)

        x4 = self.conv1x1(x3)
        x4 = self.avg(x4)
        # x = self.model(x)
        # x = x.view(-1, 256)
        # x = self.fc(x)
        return x1,x2,x3,x4
#
# class MobileNetV1(nn.Module):
#     def __init__(self):
#         super(MobileNetV1, self).__init__()
#         self.stage1 = nn.Sequential(
#             conv_bn(3, 16, 2),    # 3
#             conv_dw(16, 16, 1),   # 7
#             # conv_bn1X1(16, 16, 1),   # 7
#             conv_dw(16, 32, 2),  # 11
#             conv_dw(32, 32, 1),  # 19
#             conv_dw(32, 64, 2),  # 27
#             conv_dw(64, 64, 1),  # 43
#             conv_dw(64, 64, 1),  # 43
#             conv_dw(64, 64, 1),  # 43
#         )
#         self.stage2 = nn.Sequential(
#             conv_dw(64, 128, 2),  # 43 + 16 = 59
#             conv_dw(128, 128, 1), # 59 + 32 = 91
#             conv_dw(128, 128, 1), # 91 + 32 = 123
#         )
#         self.stage3 = nn.Sequential(
#             conv_dw(128, 128, 2), # 219 +3 2 = 241
#             conv_dw(128, 128, 1), # 241 + 64 = 301
#         )
#         self.avg = nn.AdaptiveAvgPool2d((1,1))
#         self.fc = nn.Linear(256, 1000)
#
#     def forward(self, x):
#         x = self.stage1(x)
#         x = self.stage2(x)
#         x = self.stage3(x)
#         x = self.avg(x)
#         # x = self.model(x)
#         x = x.view(-1, 256)
#         x = self.fc(x)
#         return x