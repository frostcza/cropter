#ifndef __MY_CUDA_CONVERT_H__
#define __MY_CUDA_CONVERT_H__

#include <jetson-utils/cudaUtility.h>

// Normalize IMAGE_RGB32F(0, 255) to (0.0f, 1.0f)
// Then change the data form Packed mode to Planner mode (rgbrgbrgbrgb --> rrrrggggbbbb)
// Change datatype from uchar3 to float
cudaError_t cudaPackedToPlanner(uchar3* input, size_t width, size_t height, float* output);

// change the data form Planner mode to Packed mode (rrrrggggbbbb --> rgbrgbrgbrgb)
cudaError_t cudaPlannerToPacked(float* input, size_t width, size_t height, float3* output);

#endif