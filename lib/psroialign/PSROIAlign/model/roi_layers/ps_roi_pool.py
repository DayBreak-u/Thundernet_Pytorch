import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from psroialign.PSROIAlign.model import _C


class _PSROIPool(Function):
    @staticmethod
    def forward(ctx, features, rois, pooled_height, pooled_width, spatial_scale, group_size, output_dim):
        ctx.pooled_height = int(pooled_height)
        ctx.pooled_width = int(pooled_width)
        ctx.spatial_scale = float(spatial_scale)
        ctx.group_size = int(group_size)
        ctx.output_dim = int(output_dim)
        num_rois = rois.size()[0]
        output = torch.zeros(num_rois, ctx.output_dim, ctx.pooled_height, ctx.pooled_width).to(features.device)
        mappingchannel = torch.IntTensor(num_rois, ctx.output_dim, ctx.pooled_height, ctx.pooled_width).zero_().to(features.device)
        _C.ps_roi_pool_forward(ctx.pooled_height,
                               ctx.pooled_width,
                               ctx.spatial_scale,
                               ctx.group_size,
                               ctx.output_dim,
                               features,
                               rois,
                               output,
                               mappingchannel)
        ctx.save_for_backward(rois, mappingchannel)
        # ctx.output = output
        # ctx.mappingchannel = mappingchannel
        # ctx.rois = rois
        ctx.feature_size = features.size()

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        assert(ctx.feature_size is not None and grad_output.is_cuda)
        batch_size, num_channels, data_height, data_width = ctx.feature_size
        [rois, mappingchannel] = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.zeros(batch_size, num_channels, data_height, data_width).to(grad_output.device)
            _C.ps_roi_pool_backward(ctx.pooled_height,
                                    ctx.pooled_width,
                                    ctx.spatial_scale,
                                    ctx.output_dim,
                                    grad_output,
                                    rois,
                                    grad_input,
                                    mappingchannel)
        return grad_input, None, None, None, None, None, None


ps_roi_pool = _PSROIPool.apply


class PSROIPool(nn.Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale, group_size, output_dim):
        super(PSROIPool, self).__init__()

        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        self.group_size = int(group_size)
        self.output_dim = int(output_dim)

    def forward(self, features, rois):
        return ps_roi_pool(features,
                           rois,
                           self.pooled_height,
                           self.pooled_width,
                           self.spatial_scale,
                           self.group_size,
                           self.output_dim)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "pooled_width=" + str(self.pooled_width)
        tmpstr += ", pooled_height=" + str(self.pooled_height)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", group_size=" + str(self.group_size)
        tmpstr += ", output_dim=" + str(self.output_dim)
        tmpstr += ")"
        return tmpstr
