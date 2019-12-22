from torch import  nn
from utils import  load_model
import torch.nn.functional as F
import torch
class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self,n_classes
        ):
        self.n_classes = n_classes
        super(_fasterRCNN, self).__init__()

        c_in = 1024

        self.RCNN_top = nn.Sequential(nn.Linear(5 * 7 * 7, c_in),
                                          nn.ReLU(inplace=True))


        self.RCNN_cls_score = nn.Linear(c_in, self.n_classes)
        self.RCNN_bbox_pred = nn.Linear(c_in, 4 * self.n_classes)




    def forward(self, pool5):

        pool5_flat = pool5.view(pool5.size(0), -1)
        print(pool5_flat.shape)
        fc7 = self.RCNN_top(pool5_flat)


        RCNN_cls_score = self.RCNN_cls_score(fc7)

        cls_prob = F.softmax(RCNN_cls_score, 1)

        bbox_pred =  self.RCNN_bbox_pred(fc7)


        return [cls_prob,bbox_pred]



net  = _fasterRCNN(21)

net = load_model(net, "../snet_146_3/snet_146/pascal_voc_0712/thundernet_epoch_4.pth")
net.eval()
print('Finished loading model!')
print(net)
device = torch.device("cpu")
net = net.to(device)

##################export###############
output_onnx = 'thundernet146_rcnn_head.onnx'
print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
input_names = ["roi_pool"]
# output_names = ["hm" , "wh"  , "reg"]
output_names = ["cls_prob" , "bbox_pred" ]
inputs = torch.randn(1, 5 , 7 , 7).to(device)
torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False,
                               input_names=input_names, output_names=output_names)