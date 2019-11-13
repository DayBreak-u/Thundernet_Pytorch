from .modules import *

from config import Configs
from versions.faster_rcnn.rpn import AnchorGenerator
from versions.psroialign.pollers import PsRoIAlign
from versions.faster_rcnn.faster_rcnn import FasterRCNN

CEM_FILTER = Configs.get("CEM_FILTER")
representation_size = Configs.get("representation_size")
num_classes = Configs.get("num_classes")
Snet_version = Configs.get("Snet_version")
anchor_sizes = Configs.get("anchor_sizes")
aspect_ratios = Configs.get("aspect_ratios")
Multi_size = Configs.get("Multi_size")
rpn_dense = Configs.get("rpn_dense")




class SNet(nn.Module):
    cfg = {
        49: [24, 60, 120, 240, 512],
        146: [24, 132, 264, 528],
        535: [48, 248, 496, 992],
    }

    def __init__(self,  version=49, **kwargs):
        super(SNet,self).__init__()
        num_layers = [4, 8, 4]
        self.num_layers = num_layers
        channels = self.cfg[version]
        self.channels = channels

        self.conv1 = conv_bn(
            3, channels[0], kernel_size=3, stride=2,pad = 1
        )
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1,
        )

        self.stage1 = self._make_layer(
            num_layers[0], channels[0], channels[1], **kwargs)
        self.stage2 = self._make_layer(
            num_layers[1], channels[1], channels[2], **kwargs)
        self.stage3 = self._make_layer(
            num_layers[2], channels[2], channels[3], **kwargs)
        if len(self.channels) == 5:
            self.conv5 = conv_bn(
                channels[3], channels[4], kernel_size=1, stride=1 ,pad=0 )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(channels[-1], num_classes)

    def _make_layer(self, num_layers, in_channels, out_channels, **kwargs):
        layers = [DownBlock(in_channels, out_channels, **kwargs)]
        for i in range(num_layers - 1):
            layers.append(BasicBlock(out_channels, **kwargs))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        c3 = self.stage1(x)
        c4 = self.stage2(c3)
        c5 = self.stage3(c4)
        if len(self.channels) == 5:
            c5 = self.conv5(c5)

        Cglb_lat = self.avgpool(c5)
        # Cglb_lat = Cglb_lat.view(-1, self.channels[-1], 1, 1)

        # x = self.fc(x)

        return c4,c5,Cglb_lat



class MLPHead(torch.nn.Module):
    """
    Standard heads for FPN-based models

    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(MLPHead, self).__init__()

        self.fc6 = torch.nn.Linear(in_channels, representation_size)


    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))

        return x


class FastRCNNPredictor(torch.nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = torch.nn.Linear(in_channels, num_classes)
        self.bbox_pred = torch.nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas




def get_thundernet():

    anchor_generator = AnchorGenerator(sizes=anchor_sizes,
                                       aspect_ratios=aspect_ratios)
    backbone = SNet(Snet_version)

    if len(backbone.channels) == 5:
        backbone = CEM(backbone.channels[-3] , backbone.channels[-1] ,backbone.channels[-1], backbone)
    else:
        backbone = CEM(backbone.channels[-2], backbone.channels[-1], backbone.channels[-1], backbone)

    # backbone = LightHead(CEM_SAM,CEM_FILTER,CEM_FILTER)
    backbone.out_channels = CEM_FILTER

    rpn_head = RPN(CEM_FILTER, rpn_dense)
    sam_model = SAM(rpn_dense , CEM_FILTER)



    roi_pooler = PsRoIAlign( output_size=7, sampling_ratio=2)
    onenlpHead = MLPHead(CEM_FILTER, representation_size)
    box_predictor = FastRCNNPredictor(
        representation_size,
        num_classes)

    model = FasterRCNN(backbone, num_classes=None, rpn_anchor_generator=anchor_generator, Multi_size=Multi_size ,
                       box_roi_pool=roi_pooler, rpn_head=rpn_head, sam_model = sam_model,
                       box_head=onenlpHead, box_predictor=box_predictor)

    return model
