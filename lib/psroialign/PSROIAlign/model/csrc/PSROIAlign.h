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


int PSROIAlign_forward(
    at::Tensor bottom_data,
    at::Tensor bottom_rois,
    at::Tensor top_data,
    at::Tensor argmax_data,
    float spatial_scale,
    int group_size,
    int sampling_ratio) {

#ifdef WITH_CUDA
    CHECK_INPUT(bottom_data);
    CHECK_INPUT(bottom_rois);
    CHECK_INPUT(top_data);
    CHECK_INPUT(argmax_data);

    int size_rois = bottom_rois.size(1);

    if (size_rois != 5) {
        printf("wrong roi size. (roi size should be 5)\n");
        return 0;
    }

    cudaStream_t stream = THCState_getCurrentStream(state);

    PSROIAlignForwardLaucher(bottom_data,
                             bottom_rois,
                             top_data,
                             argmax_data,
                             spatial_scale,
                             group_size,
                             sampling_ratio,
                             stream);
#endif
    return 1;
}

int PSROIAlign_backward(
    at::Tensor top_diff,
    at::Tensor argmax_data,
    at::Tensor bottom_rois,
    at::Tensor bottom_diff,
    float spatial_scale,
    int group_size,
    int sampling_ratio) {

#ifdef WITH_CUDA
    CHECK_INPUT(top_diff);
    CHECK_INPUT(bottom_rois);
    CHECK_INPUT(bottom_diff);
    CHECK_INPUT(argmax_data);

    int size_rois = bottom_rois.size(1);

    if (size_rois != 5) {
        printf("wrong roi size. (roi size should be 5)\n");
        return 0;
    }

    cudaStream_t stream = THCState_getCurrentStream(state);

    PSROIAlignBackwardLaucher(top_diff,
                              argmax_data,
                              bottom_rois,
                              bottom_diff,
                              spatial_scale,
                              group_size,
                              sampling_ratio,
                              stream);
#endif
    return 1;
}
