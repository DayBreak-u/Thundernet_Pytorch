from psroialign.PSROIAlign.model.roi_layers import PSROIAlign,PSROIPool
from torch import  nn

class PSROIAlignhandle(nn.Module):
    def __init__(self,
                 spatial_scale=1./16.,
                 roi_size=7,
                 sampling_ratio=2,
                 pooled_dim=5):

        super(PSROIAlignhandle, self).__init__()
        self.psroialign = PSROIAlign(spatial_scale=spatial_scale,
                                     roi_size=roi_size,
                                     sampling_ratio=sampling_ratio,
                                     pooled_dim=pooled_dim)

    def forward(self, feat, rois):
        # print(feat.shape)
        pooled_feat = self.psroialign(feat, rois)

        return pooled_feat



class PSROIPoolhandle(nn.Module):
    def __init__(self,
                 pooled_height=7,
                 pooled_width=7,
                 spatial_scale=1./16.,
                 group_size=7,
                 output_dim=5):

        super(PSROIPoolhandle, self).__init__()
        self.psroipool = PSROIPool(pooled_height=pooled_height,
                                   pooled_width=pooled_width,
                                   spatial_scale=spatial_scale,
                                   group_size=group_size,
                                   output_dim=output_dim)

    def forward(self, feat, rois):
        pooled_feat = self.psroipool(feat, rois)
        return pooled_feat