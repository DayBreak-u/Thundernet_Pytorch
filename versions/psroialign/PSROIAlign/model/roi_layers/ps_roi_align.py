from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import torch
from versions.psroialign.PSROIAlign.model import _C


class _PSROIAlign(Function):
    @staticmethod
    def forward(ctx, bottom_data, bottom_rois, spatial_scale, roi_size, sampling_ratio, pooled_dim):
        ctx.spatial_scale = spatial_scale  # 1./16.
        ctx.roi_size = roi_size  # 7
        ctx.sampling_ratio = sampling_ratio  # 2
        ctx.pooled_dim = pooled_dim  # 10
        ctx.feature_size = bottom_data.size()  # (B, 490, H, W)
        num_rois = bottom_rois.size(0)  # B*K
        # (B*K, 10, 7, 7)
        top_data = torch.zeros([num_rois, pooled_dim, roi_size, roi_size], dtype=torch.float32).to(bottom_data.device)
        # (B*K, 10, 7, 7)
        argmax_data = torch.zeros([num_rois, pooled_dim, roi_size, roi_size], dtype=torch.int32).to(bottom_data.device)
        if bottom_data.is_cuda:
            _C.ps_roi_align_forward(bottom_data,    # (B, 490, H, W)
                                    bottom_rois,    # (B*K, 5), e.g. K = 128
                                    top_data,       # (B*K, 10, 7, 7)
                                    argmax_data,    # (B*K, 10, 7, 7)
                                    spatial_scale,  # 1./16.
                                    roi_size,       # 7
                                    sampling_ratio  # 2
                                    )
            ctx.save_for_backward(bottom_rois, argmax_data)
        else:
            raise NotImplementedError

        return top_data

    @staticmethod
    @once_differentiable
    def backward(ctx, top_diff):
        spatial_scale = ctx.spatial_scale  # 1./16.
        roi_size = ctx.roi_size  # 7
        sampling_ratio = ctx.sampling_ratio  # 2
        batch_size, channels, height, width = ctx.feature_size
        [bottom_rois, argmax_data] = ctx.saved_tensors
        bottom_diff = None
        if ctx.needs_input_grad[0]:
            bottom_diff = torch.zeros([batch_size, channels, height, width], dtype=torch.float32).to(top_diff.device)
            _C.ps_roi_align_backward(top_diff,      # (B*K, 10, 7, 7)
                                     argmax_data,   # (B*K, 10, 7, 7)
                                     bottom_rois,   # (B*K, 10, 7, 7)
                                     bottom_diff,   # (B, 490, H, W)
                                     spatial_scale, # 1./16.
                                     roi_size,      # 7
                                     sampling_ratio # 2
                                     )

        return bottom_diff, None, None, None, None, None


ps_roi_align = _PSROIAlign.apply


class PSROIAlign(nn.Module):
    def __init__(self, spatial_scale, roi_size, sampling_ratio, pooled_dim):
        super(PSROIAlign, self).__init__()
        self.spatial_scale = spatial_scale
        self.roi_size = roi_size
        self.sampling_ratio = sampling_ratio
        self.pooled_dim = pooled_dim

    def forward(self, bottom_data, bottom_rois):
        return ps_roi_align(bottom_data,  # (B, 490, H, W)
                            bottom_rois,  # (B*K, 5)
                            self.spatial_scale,  # 1./16.
                            self.roi_size,  # 7
                            self.sampling_ratio,  # 2
                            self.pooled_dim  # 10
                            )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", roi_size=" + str(self.roi_size)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ", pooled_dim=" + str(self.pooled_dim)
        tmpstr += ")"
        return tmpstr
