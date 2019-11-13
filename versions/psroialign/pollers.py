# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn


from versions.psroialign.psroialign import PSROIAlignhandle , PSROIPoolhandle
from config import  Configs

CEM_FILTER = Configs.get("CEM_FILTER")
spatial_scale = Configs.get("spatial_scale")





class PsRoIAlign(nn.Module):
    """
    Multi-scale RoIAlign pooling, which is useful for detection with or without FPN.

    It infers the scale of the pooling via the heuristics present in the FPN paper.

    Arguments:
        featmap_names (List[str]): the names of the feature maps that will be used
            for the pooling.
        output_size (List[Tuple[int, int]] or List[int]): output size for the pooled region
        sampling_ratio (int): sampling ratio for ROIAlign

    Examples::

    """

    def __init__(self,  output_size, sampling_ratio):
        super(PsRoIAlign, self).__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)

        self.sampling_ratio = sampling_ratio
        self.output_size = tuple(output_size)
        self.scales = spatial_scale


    def convert_to_roi_format(self, boxes):
        concat_boxes = torch.cat(boxes, dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = torch.cat(
            [
                torch.full((len(b), 1), i, dtype=dtype, device=device)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois


    def forward(self, x, boxes, image_shapes):
        """
        Arguments:
            x (OrderedDict[Tensor]): feature maps for each level. They are assumed to have
                all the same number of channels, but they can have different sizes.
            boxes (List[Tensor[N, 4]]): boxes to be used to perform the pooling operation, in
                (x1, y1, x2, y2) format and in the image reference size, not the feature map
                reference.
            image_shapes (List[Tuple[height, width]]): the sizes of each image before they
                have been fed to a CNN to obtain feature maps. This allows us to infer the
                scale factor for each one of the levels to be pooled.
        Returns:
            result (Tensor)
        """


        rois = self.convert_to_roi_format(boxes)


        roi_align = PSROIAlignhandle(sampling_ratio=self.sampling_ratio, spatial_scale=self.scales, roi_size=7,
                                      pooled_dim=CEM_FILTER//(7*7))


        return roi_align(
            x, rois
        )


