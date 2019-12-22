import  sys
sys.path.insert(0,"../lib")
from  torch import  nn
from model.faster_rcnn.modules import  *
from model.faster_rcnn.Snet import SnetExtractor
from utils import  load_model
class _RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din):
        super(_RPN, self).__init__()

        self.din = din  # get depth of input feature map, e.g., 512


        # define the convrelu layers processing input feature map
        # self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        # define bg/fg classifcation score layer
        self.nc_score_out = 25*2
        self.RPN_cls_score = nn.Conv2d(self.din, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction layer
        self.nc_bbox_out = 25 * 4  # 4(coords) * 9 (anchors)
        self.RPN_bbox_pred = nn.Conv2d(self.din, self.nc_bbox_out, 1, 1, 0)
        self.softmax  = nn.Softmax(1)

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(input_shape[0], int(d),
                   int(float(input_shape[1] * input_shape[2]) / float(d)),
                   input_shape[3])
        return x

    def forward(self, base_feat):

        rpn_cls_score = self.RPN_cls_score(base_feat)

        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = self.softmax(rpn_cls_score_reshape)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        # get rpn offsets to the anchor boxes
        rpn_bbox_pred = self.RPN_bbox_pred(base_feat)


        return rpn_cls_prob, rpn_bbox_pred



class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self
        ):
        super(_fasterRCNN, self).__init__()


        self.RCNN_base = SnetExtractor(146)

        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        # self.focalloss_handle = FocalLossV4(num_class=21, alpha=0.25, gamma=2.0, balance_index=2)
        # define Large Separable Convolution Layer

        self.rpn = RPN(in_channels=245, f_channels=256)


        self.sam = SAM(256,245)
        # define rpn
        self.RCNN_rpn = _RPN(256)




    def forward(self, im_data):

        basefeat = self.RCNN_base(im_data)

        # feed base feature map tp RPN to obtain rois
        rpn_feat= self.rpn(basefeat)

        rpn_cls_prob, rpn_bbox_pred = self.RCNN_rpn(rpn_feat)

        base_feat = self.sam([basefeat,rpn_feat])
        return [rpn_cls_prob, rpn_bbox_pred ,base_feat]

net  = _fasterRCNN()

net = load_model(net, "../snet_146_3/snet_146/pascal_voc_0712/thundernet_epoch_4.pth")
net.eval()
print('Finished loading model!')
print(net)
device = torch.device("cpu")
net = net.to(device)

##################export###############
output_onnx = 'thundernet146_rpn.onnx'
print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
input_names = ["input"]
# output_names = ["hm" , "wh"  , "reg"]
output_names = ["rpn_cls_prob" , "rpn_bbox_pred" , "base_feat" ]
inputs = torch.randn(1, 3, 320, 320).to(device)
torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False,
                               input_names=input_names, output_names=output_names)