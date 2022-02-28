#include "myCudaResize.h"

__device__ void gpuBilinear(float p_x, float p_y, uchar3* input, int iWidth, int iHeight, float3* output, int index)
{
    const int c00_x = (int)floor(p_x);
    const int c00_y = (int)floor(p_y);
    const float t_x = p_x - c00_x;
    const float t_y = p_y - c00_y;
    uchar3 c00 = input[c00_y * iWidth + c00_x];
    uchar3 c01 = make_uchar3(128, 128, 128);
    uchar3 c10 = make_uchar3(128, 128, 128);
    uchar3 c11 = make_uchar3(128, 128, 128);

    if(c00_x + 1 < iWidth && c00_y + 1 < iHeight)
    {
        c01 = input[(c00_y+1) * iWidth + c00_x];
        c10 = input[c00_y * iWidth + c00_x+1];
        c11 = input[(c00_y+1) * iWidth + c00_x+1];
    }
    // p = c00*(1-tx)*(1-ty) + c10*tx*(1-ty) + c01*(1-tx)*ty + c11*tx*ty
    output[index].x = c00.x*(1.0-t_x)*(1.0-t_y) + c10.x*t_x*(1.0-t_y) + c01.x*(1.0-t_x)*t_y + c11.x*t_x*t_y;
    output[index].y = c00.y*(1.0-t_x)*(1.0-t_y) + c10.y*t_x*(1.0-t_y) + c01.y*(1.0-t_x)*t_y + c11.y*t_x*t_y;
    output[index].z = c00.z*(1.0-t_x)*(1.0-t_y) + c10.z*t_x*(1.0-t_y) + c01.z*(1.0-t_x)*t_y + c11.z*t_x*t_y;
}

__global__ void gpuResizeNoStretch( float2 scale, uchar3* input, int iWidth, int iHeight, float3* output, int oWidth, int oHeight )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= oWidth || y >= oHeight )
		return;

    if(scale.y > scale.x)
    {
        const int border = (oHeight - (int)(scale.x * iHeight)) / 2;
        if(y < border || y >= oHeight - border)
        {
            output[y*oWidth+x] = make_float3(128.0, 128.0, 128.0);
        }
        else
        {
            /* old version without bilinear interpolation
                const int dx = ((float)x / scale.x);
                const int dy = ((float)(y - border) / scale.x);
                output[y*oWidth+x].x = (float)input[ dy * iWidth + dx ].x;
                output[y*oWidth+x].y = (float)input[ dy * iWidth + dx ].y;
                output[y*oWidth+x].z = (float)input[ dy * iWidth + dx ].z;
            */

            const float p_x = ((float)x / scale.x);
            const float p_y = ((float)(y - border) / scale.x);
            gpuBilinear(p_x, p_y, input, iWidth, iHeight, output, y*oWidth+x);
        }
    }
    else
    {
        const int border = (oWidth - (int)(scale.y * iWidth)) / 2;
        if(x < border || x >= oWidth - border)
        {
            output[y*oWidth+x] = make_float3(128.0, 128.0, 128.0);
        }
        else
        {
            const float p_x = ((float)(x - border) / scale.y);
            const float p_y = ((float)y / scale.y);
            gpuBilinear(p_x, p_y, input, iWidth, iHeight, output, y*oWidth+x);
        }
    }
}

// Resize an unchar3 image without strentching
// For example, input_size = 1920*1080, output_size = 640*640, we first downsample input image 3x to 640*360, then embed it in 640*640
// The unused pixels are set to (128.0,128.0,128.0)
// using bilinear interpolation https://zhuanlan.zhihu.com/p/77496615
cudaError_t cudaResizeNoStretch( uchar3* input, size_t inputWidth, size_t inputHeight, float3* output, size_t outputWidth, size_t outputHeight)
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	const float2 scale = make_float2( float(outputWidth) / float(inputWidth),
							     float(outputHeight) / float(inputHeight) );

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	gpuResizeNoStretch<<<gridDim, blockDim>>>(scale, input, inputWidth, inputHeight, output, outputWidth, outputHeight);

	return CUDA(cudaGetLastError());
}


__global__ void gpuPacked2Planner(float3* input, size_t width, size_t height, float* output)
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

// Normalize IMAGE_RGB32F(0, 255) to (0.0f, 1.0f)
// Then change the Packed mode to Planner mode (rgbrgbrgbrgb --> rrrrggggbbbb)
cudaError_t cudaPacked2Planner(float3* input, size_t width, size_t height, float* output)
{
    if( !input || !output )
		return cudaErrorInvalidDevicePointer;

    if( width == 0 || height == 0)
		return cudaErrorInvalidValue;

    // launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));
    
    gpuPacked2Planner<<<gridDim, blockDim>>>(input, width, height, output);

    return CUDA(cudaGetLastError());
}


