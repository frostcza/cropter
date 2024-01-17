#include "IFCNN.h"
#include <jetson-utils/imageIO.h>
#include <jetson-utils/cudaResize.h>
#include <jetson-utils/glDisplay.h>
#include <jetson-utils/cudaColorspace.h>
#include "myCudaConvert.h"
#include <string>
#include <vector>
#include <chrono>

#include <cstdio>
#include "stb_image_write.h"

using namespace std;

int main()
{
    // const char* ir_path = "../test_ir.png";
    // const char* vi_path = "../test_vi.png";
    const char* ir_path = "/home/cza/IFCNN/c++/build_network/test_ir.jpg";
    const char* vi_path = "/home/cza/IFCNN/c++/build_network/test_vi.jpg";
    string engine_path = "../IFCNN_int8.engine";
    const char* save_path = "../result.jpg";

    void* ir = NULL;
    void* vi = NULL;
    int width = 0;
    int height = 0;
    IFCNN* ifcnn = new IFCNN(engine_path);

    void* fused_image = NULL;
    cudaMalloc((void**)&fused_image, ifcnn->INPUT_W * ifcnn->INPUT_H * sizeof(float3));

    void* converted = NULL;
    cudaMalloc((void**)&converted, ifcnn->INPUT_W * ifcnn->INPUT_H * sizeof(uchar3));

    void* ir_resized = NULL;
    void* vi_resized = NULL;
    cudaMalloc((void**)&ir_resized, ifcnn->INPUT_W * ifcnn->INPUT_H * sizeof(uchar3));
    cudaMalloc((void**)&vi_resized, ifcnn->INPUT_W * ifcnn->INPUT_H * sizeof(uchar3));

    for(int i = 0; i < 20; i++)
    {
        loadImage(ir_path, &ir, &width, &height, IMAGE_RGB8);
        loadImage(vi_path, &vi, &width, &height, IMAGE_RGB8);
        cudaResize((uchar3*) ir, width, height, (uchar3*) ir_resized, ifcnn->INPUT_W, ifcnn->INPUT_H);
        cudaResize((uchar3*) vi, width, height, (uchar3*) vi_resized, ifcnn->INPUT_W, ifcnn->INPUT_H);
        auto start = std::chrono::high_resolution_clock::now();
        ifcnn->doInference((uchar3*)ir_resized, (uchar3*)vi_resized, fused_image);
        auto end = std::chrono::high_resolution_clock::now();
        int time_elapse = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << time_elapse << "ms" << std::endl;
    }

    // 不知道为什么直接用SaveImage()存图会segmentation fault，gdb看到调了stbi_write()
    // 但把数据拷到cpu上，写txt发现数据没问题，stbi_write_jpg也能存下来

    // float im[ifcnn->INPUT_W * ifcnn->INPUT_H * 3];
    // cudaMemcpy(im, fused_image, ifcnn->INPUT_W * ifcnn->INPUT_H * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    // float x;
    // for(int i = 0 ; i < ifcnn->INPUT_W * ifcnn->INPUT_H * 3; i++)
    // {
    //     x = im[i];
    //     std::cout << x << " ";
    // }

    unsigned char im[ifcnn->INPUT_W * ifcnn->INPUT_H * 3];
    cudaMemcpy(im, fused_image, ifcnn->INPUT_W * ifcnn->INPUT_H * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // FILE* file = fopen("1.txt","w");
    // if(!file) return -1;
    // unsigned char x;
    // for(int i = 0 ; i < ifcnn->INPUT_W * ifcnn->INPUT_H * 3; i++)
    // {
    //     x = im[i];
    //     printf("%d ", x);
    //     fprintf(file,"%d ",x);
    // }
    // fclose(file);

    stbi_write_jpg(save_path, ifcnn->INPUT_W, ifcnn->INPUT_H, 3, (void*)im, 95);

    // cudaMemcpy(converted, im, ifcnn->INPUT_W * ifcnn->INPUT_H * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    // cudaDeviceSynchronize();
    // saveImage(save_path, converted, ifcnn->INPUT_W, ifcnn->INPUT_H, IMAGE_RGB8);

    cudaFreeHost(fused_image);
    cudaFreeHost(converted);
    cudaFreeHost(ir_resized);
    cudaFreeHost(vi_resized);

    SAFE_DELETE(ifcnn);

    return 0;
}