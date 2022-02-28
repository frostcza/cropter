#ifndef __MY_CUDA_RESIZE_H__
#define __MY_CUDA_RESIZE_H__

#include <jetson-utils/cudaUtility.h>

cudaError_t cudaResizeNoStretch( uchar3* input, size_t inputWidth, size_t inputHeight, float3* output, size_t outputWidth, size_t outputHeight);
cudaError_t cudaPacked2Planner(float3* input, size_t width, size_t height, float* output);

#endif