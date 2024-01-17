#include "myCudaConvert.h"

__global__ void gpuPackedToPlanner(uchar3* input, size_t width, size_t height, float* output)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if( x >= width || y >= height )
        return;

    const int pixel = y * width + x;

    output[pixel] = (float)input[pixel].x / 255.0f; // red
    output[pixel + width * height] = (float)input[pixel].y / 255.0f; // green
    output[pixel + width * height * 2] = (float)input[pixel].z / 255.0f; // blue

}

cudaError_t cudaPackedToPlanner(uchar3* input, size_t width, size_t height, float* output)
{
    if( !input || !output )
        return cudaErrorInvalidDevicePointer;

    if( width == 0 || height == 0)
        return cudaErrorInvalidValue;

    // launch kernel
    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));
    
    gpuPackedToPlanner<<<gridDim, blockDim>>>(input, width, height, output);

    return CUDA(cudaGetLastError());
}

__global__ void gpuPlannerToPacked(float* input, size_t width, size_t height, float3* output)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if( x >= width || y >= height )
        return;

    const int pixel = y * width + x;

    output[pixel].x = input[pixel];
    output[pixel].y = input[pixel + width * height];
    output[pixel].z = input[pixel + width * height * 2];
    if(output[pixel].x > 1.0f) output[pixel].x = 1.0f;
    if(output[pixel].x < 0.0f) output[pixel].x = 0.0f;
    if(output[pixel].y > 1.0f) output[pixel].y = 1.0f;
    if(output[pixel].y < 0.0f) output[pixel].y = 0.0f;
    if(output[pixel].z > 1.0f) output[pixel].z = 1.0f;
    if(output[pixel].z < 0.0f) output[pixel].z = 0.0f;

    // output[pixel].x *= 255;
    // output[pixel].y *= 255;
    // output[pixel].z *= 255;
}

cudaError_t cudaPlannerToPacked(float* input, size_t width, size_t height, float3* output)
{
    if( !input || !output )
        return cudaErrorInvalidDevicePointer;

    if( width == 0 || height == 0)
        return cudaErrorInvalidValue;

    // launch kernel
    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));
    
    gpuPlannerToPacked<<<gridDim, blockDim>>>(input, width, height, output);

    return CUDA(cudaGetLastError());
}