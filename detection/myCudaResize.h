#ifndef __MY_CUDA_RESIZE_H__
#define __MY_CUDA_RESIZE_H__

#include <jetson-utils/cudaUtility.h>

// Resize an unchar3 image without strentching
// For example, input_size = 1920*1080, output_size = 640*640, we first downsample the input image 3x to 640*360, then embed it in 640*640
// The unused pixels are set to (128.0,128.0,128.0)
// using bilinear interpolation https://zhuanlan.zhihu.com/p/77496615
cudaError_t cudaResizeNoStretch( uchar3* input, size_t inputWidth, size_t inputHeight, float3* output, size_t outputWidth, size_t outputHeight);

// Normalize IMAGE_RGB32F(0, 255) to (0.0f, 1.0f)
// Then change the data form Packed mode to Planner mode (rgbrgbrgbrgb --> rrrrggggbbbb)
cudaError_t cudaPacked2Planner(float3* input, size_t width, size_t height, float* output);

#endif