// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "PSROIAlign.h"
#include "PSROIPool.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ps_roi_align_forward", &PSROIAlign_forward, "PSROIAlign_forward");
  m.def("ps_roi_align_backward", &PSROIAlign_backward, "PSROIAlign_backward");
  m.def("ps_roi_pool_forward", &PSROIPool_forward, "PSROIPool_forward");
  m.def("ps_roi_pool_backward", &PSROIPool_backward, "PSROIPool_backward");
}
