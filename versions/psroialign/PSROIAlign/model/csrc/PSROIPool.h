// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
#include <torch/extension.h>

#ifdef WITH_CUDA
#include "cuda/vision.h"

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

extern THCState* state;

#endif


int PSROIPool_forward(int pooled_height,
                      int pooled_width,
                      float spatial_scale,
                      int group_size,
                      int output_dim,
                      at::Tensor features,
                      at::Tensor rois,
                      at::Tensor output,
                      at::Tensor mappingchannel) {
#ifdef WITH_CUDA
    CHECK_INPUT(features);
    CHECK_INPUT(rois);
    CHECK_INPUT(output);
    CHECK_INPUT(mappingchannel);

	// Get # of Rois
	int num_rois = rois.size(0);
	int size_rois = rois.size(1);
	if (size_rois != 5) {
        printf("wrong roi size\n");
		return 0;
	}

	int data_height = features.size(2);
	int data_width = features.size(3);
	int num_channels = features.size(1);

	cudaStream_t stream = THCState_getCurrentStream(state);

	// call the gpu kernel for psroi_pooling
	PSROIPoolForwardLauncher(features,
	                         spatial_scale,
	                         num_rois,
	                         data_height,
	                         data_width,
	                         num_channels,
	                         pooled_height,
	                         pooled_width,
	                         rois,
	                         group_size,
	                         output_dim,
	                         output,
	                         mappingchannel,
	                         stream);
#endif
	return 1;
}


int PSROIPool_backward(int pooled_height,
                       int pooled_width,
                       float spatial_scale,
                       int output_dim,
                       at::Tensor top_grad,
                       at::Tensor rois,
                       at::Tensor bottom_grad,
                       at::Tensor mappingchannel) {
#ifdef WITH_CUDA
    CHECK_INPUT(top_grad);
    CHECK_INPUT(rois);
    CHECK_INPUT(bottom_grad);
    CHECK_INPUT(mappingchannel);

    int batch_size = bottom_grad.size(0);

    // Number of ROIs
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);
    if (size_rois != 5) {
        	return 0;
    }

    // data height
    int data_height = bottom_grad.size(2);
    // data width
    int data_width = bottom_grad.size(3);
    // Number of channels
    int num_channels = bottom_grad.size(1);

    cudaStream_t stream = THCState_getCurrentStream(state);

    PSROIPoolBackwardLauncher(top_grad,
                              mappingchannel,
                              batch_size,
                              num_rois,
                              spatial_scale,
                              num_channels,
                              data_height,
                              data_width,
                              pooled_width,
                              pooled_height,
                              output_dim,
                              bottom_grad,
                              rois,
                              stream);
#endif
    return 1;
}
