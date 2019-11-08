from versions.psroialign.PSROIAlign.model.roi_layers import PSROIAlign
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



