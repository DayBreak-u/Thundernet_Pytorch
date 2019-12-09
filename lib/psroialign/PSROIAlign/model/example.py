import torch
import torch.nn as nn
from roi_layers import PSROIAlign

#
# class PSROIPoolExample(nn.Module):
#     def __init__(self,
#                  pooled_height=7,
#                  pooled_width=7,
#                  spatial_scale=1./16.,
#                  group_size=7,
#                  output_dim=10):
#
#         super(PSROIPoolExample, self).__init__()
#         self.psroipool = PSROIPool(pooled_height=pooled_height,
#                                    pooled_width=pooled_width,
#                                    spatial_scale=spatial_scale,
#                                    group_size=group_size,
#                                    output_dim=output_dim)
#
#     def forward(self, feat, rois):
#         print("PSROIPool:")
#         print(f"feature.shape:\t{feat.shape}")
#         print(f"rois.shape:\t{rois.shape}")
#         pooled_feat = self.psroipool(feat, rois)
#         print(f"pooled feature: {pooled_feat.shape}\n{pooled_feat}\n")
#         return pooled_feat


class PSROIAlignExample(nn.Module):
    def __init__(self,
                 spatial_scale=1./16.,
                 roi_size=7,
                 sample_ratio=2,
                 pooled_dim=10):

        super(PSROIAlignExample, self).__init__()
        self.psroialign = PSROIAlign(spatial_scale=spatial_scale,
                                     roi_size=roi_size,
                                     sampling_ratio=sample_ratio,
                                     pooled_dim=pooled_dim)

    def forward(self, feat, rois):
        print("PSROIAlign:")
        print(f"feature.shape:\t{feat.shape}")
        print(f"rois.shape:\t{rois.shape}")
        pooled_feat = self.psroialign(feat, rois)
        print(f"pooled feature: {pooled_feat.shape}\n{pooled_feat}\n")
        return pooled_feat


if __name__ == '__main__':
    if not torch.cuda.is_available():
        exit('Only works with cuda')

    # psroipool_example = PSROIPoolExample()
    psroialign_example = PSROIAlignExample()

    # feature map to be pooled
    batch_size = 4
    feat_height = 30
    feat_width = 40
    roi_size = 7
    oup_dim = 10

    feature = torch.randn((batch_size,
                           roi_size * roi_size * oup_dim,
                           feat_height,
                           feat_width),
                          requires_grad=True).cuda()

    # RoI: (batch_index, x1, y1, x2, y2)
    rois = torch.tensor([
        [0, 1., 1., 5., 5.],
        [0, 3., 3., 9., 9.],
        [1, 5., 5., 10., 10.],
        [1, 7., 7., 12., 12.]
    ]).cuda()

    # PSROIPool and PSROIAlign
    # psroipool_pooled_feat = psroipool_example(feature, rois)
    psroialign_pooled_feat = psroialign_example(feature, rois)
