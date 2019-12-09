#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


__global__ void PSROIPoolForward(
	const int nthreads,  		// (B*K) * 10 * 7 * 7
	const float* __restrict__ bottom_data,	// (B, 490, H, W)
	const float spatial_scale, 	// 1./16.
	const int height, 			// H
	const int width,			// W
	const int channels, 		// 490
	const int pooled_height, 	// 7
	const int pooled_width,		// 7
	const int group_size, 		// 7
	const int output_dim,		// 10
	const float* __restrict__ bottom_rois, 	// (B*K, 5)
	float* __restrict__ top_data, 			// (B*K, 10, 7, 7)
	int* __restrict__ mapping_channel		// (B*K, 10, 7, 7)
) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
		/* (n, ctop, ph, pw) is an element in the pooled output.
		 * Whole size is up to (B*K, 10, 7, 7), where
		 * n is up to B*K, e.g. K = 128,
		 * ctop is up to 10,
		 * ph is up to 7
		 * pw is up to 7
		 */ 
        int pw = index % pooled_width;
      	int ph = (index / pooled_width) % pooled_height;
      	int ctop = (index / pooled_width / pooled_height) % output_dim;
      	int n = index / pooled_width / pooled_height / output_dim;

        bottom_rois += n * 5;
        int roi_batch_ind = bottom_rois[0];
		float roi_start_w = static_cast<float>(round(bottom_rois[1])) * spatial_scale;
      	float roi_start_h = static_cast<float>(round(bottom_rois[2])) * spatial_scale;
      	float roi_end_w = static_cast<float>(round(bottom_rois[3]) + 1.) * spatial_scale;
      	float roi_end_h = static_cast<float>(round(bottom_rois[4]) + 1.) * spatial_scale;

        // Force malformed ROIs to be 1x1
        float roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0
      	float roi_height = max(roi_end_h - roi_start_h, 0.1);

        float bin_size_h = (float)(roi_height) / (float)(pooled_height);
        float bin_size_w = (float)(roi_width) / (float)(pooled_width);

        int hstart = floor(static_cast<float>(ph) * bin_size_h + roi_start_h);
      	int wstart = floor(static_cast<float>(pw)* bin_size_w + roi_start_w);
      	int hend = ceil(static_cast<float>(ph + 1) * bin_size_h + roi_start_h);
      	int wend = ceil(static_cast<float>(pw + 1) * bin_size_w + roi_start_w);

        // Add roi offsets and clip to input boundaries
        hstart = min(max(hstart, 0), height);
      	hend = min(max(hend, 0), height);
      	wstart = min(max(wstart, 0), width);
      	wend = min(max(wend, 0), width);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        int gw = pw;
      	int gh = ph;
      	int c = (ctop * group_size + gh) * group_size + gw;

        bottom_data += (roi_batch_ind * channels + c) * height * width;
        float out_sum = 0;
      	for (int h = hstart; h < hend; ++h) {
      	  for (int w = wstart; w < wend; ++w) {
      	    int bottom_index = h * width + w;
      	    out_sum += bottom_data[bottom_index];
      	  }
      	}
      	float bin_area = (hend - hstart) * (wend - wstart);
      	top_data[index] = is_empty ? 0. : out_sum / bin_area;
      	mapping_channel[index] = c;
    }
}


int PSROIPoolForwardLauncher(
	at::Tensor bottom_data,  	// (B, 490, H, W)
	const float spatial_scale, 	// 1./16.
	const int num_rois, 		// B*K, K = 128		
	const int height,			// H
	const int width, 			// W
	const int channels, 		// 490
	const int pooled_height,	// 7
	const int pooled_width, 	// 7
	at::Tensor bottom_rois,	    // (B*K, 5)
	const int group_size, 		// 7
	const int output_dim,		// 10
	at::Tensor top_data, 	    // (B*K, 10, 7, 7)
	at::Tensor mapping_channel, // (B*K, 10, 7, 7)
	cudaStream_t stream
) {

    const int kThreadsPerBlock = 1024;
    const int output_size = output_dim * pooled_height * pooled_width * num_rois;

    PSROIPoolForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
		output_size, 
		bottom_data.data<float>(),
		spatial_scale, 
		height,
		width,
		channels, 
		pooled_height,
		pooled_width, 
		group_size, 
		output_dim, 
		bottom_rois.data<float>(),
		top_data.data<float>(),
		mapping_channel.data<int>());

	cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err) {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }
    return 1;
}


__global__ void PSROIPoolBackward(const int nthreads, const float* __restrict__ top_diff,
    const int* __restrict__ mapping_channel, const int num_rois, const float spatial_scale,
    const int height, const int width, const int channels,
    const int pooled_height, const int pooled_width, const int output_dim, float* __restrict__ bottom_diff,
    const float* __restrict__ bottom_rois) {
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {

      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int n = index / pooled_width / pooled_height / output_dim;

      // [start, end) interval for spatial sampling
      bottom_rois += n * 5;
      int roi_batch_ind = bottom_rois[0];
      float roi_start_w =
        static_cast<float>(round(bottom_rois[1])) * spatial_scale;
      float roi_start_h =
        static_cast<float>(round(bottom_rois[2])) * spatial_scale;
      float roi_end_w =
        static_cast<float>(round(bottom_rois[3]) + 1.) * spatial_scale;
      float roi_end_h =
        static_cast<float>(round(bottom_rois[4]) + 1.) * spatial_scale;

      // Force too small ROIs to be 1x1
      float roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0
      float roi_height = max(roi_end_h - roi_start_h, 0.1);

      // Compute w and h at bottom
      float bin_size_h = roi_height / static_cast<float>(pooled_height);
      float bin_size_w = roi_width / static_cast<float>(pooled_width);

      int hstart = floor(static_cast<float>(ph)* bin_size_h
        + roi_start_h);
      int wstart = floor(static_cast<float>(pw)* bin_size_w
        + roi_start_w);
      int hend = ceil(static_cast<float>(ph + 1) * bin_size_h
        + roi_start_h);
      int wend = ceil(static_cast<float>(pw + 1) * bin_size_w
        + roi_start_w);
      // Add roi offsets and clip to input boundaries
      hstart = min(max(hstart, 0), height);
      hend = min(max(hend, 0), height);
      wstart = min(max(wstart, 0), width);
      wend = min(max(wend, 0), width);
      bool is_empty = (hend <= hstart) || (wend <= wstart);

      // Compute c at bottom
      int c = mapping_channel[index];
      float* offset_bottom_diff = bottom_diff +
        (roi_batch_ind * channels + c) * height * width;
      float bin_area = (hend - hstart)*(wend - wstart);
      float diff_val = is_empty ? 0. : top_diff[index] / bin_area;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          int bottom_index = h*width + w;
          //caffe_gpu_atomic_add(diff_val, offset_bottom_diff + bottom_index);
          atomicAdd(offset_bottom_diff + bottom_index, diff_val);
        }
      }
  }
}

int PSROIPoolBackwardLauncher(at::Tensor top_diff,
                              at::Tensor mapping_channel,
                              const int batch_size,
                              const int num_rois,
                              const float spatial_scale,
                              const int channels,
                              const int height,
                              const int width,
                              const int pooled_width,
                              const int pooled_height,
                              const int output_dim,
                              at::Tensor bottom_diff,
                              at::Tensor bottom_rois,
                              cudaStream_t stream) {

    const int kThreadsPerBlock = 1024;
    //const int output_size = output_dim * height * width * channels;
    const int output_size = output_dim * pooled_height * pooled_width * num_rois;

    PSROIPoolBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
      output_size,
      top_diff.data<float>(),
      mapping_channel.data<int>(),
      num_rois,
      spatial_scale,
      height,
      width,
      channels,
      pooled_height,
      pooled_width,
      output_dim,
      bottom_diff.data<float>(),
      bottom_rois.data<float>());

	cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err) {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}
