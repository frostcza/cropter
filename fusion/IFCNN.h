#ifndef __IFCNN_H__
#define __IFCNN_H__

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <jetson-utils/cudaUtility.h>
#include <jetson-utils/cudaMappedMemory.h>
#include "NvInfer.h"


using namespace nvinfer1;
using namespace std;

class IFCNN
{
public:

    IFCNN();
    IFCNN(string engine_path);
    ~IFCNN();
    bool Init();
    void doInference(uchar3* ir, uchar3* vi, void* fused_image);

    class Logger : public ILogger           
    {
        void log(Severity severity, AsciiChar const * msg) noexcept override
        {
            // suppress info-level messages
            if (severity <= Severity::kWARNING)
                printf("[IFCNN] %s\n", msg);
        }
    } gLogger;

    string engine_name;
    const int INPUT_H = 512;
    const int INPUT_W = 640;
    const int INPUT_C = 3;

    const char* INPUT_IR_NAME = "ir";
    const char* INPUT_VI_NAME = "vi";
    const char* OUTPUT_FUSED_NAME = "fused";

    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t stream;
    
    float* buffers[3];
    void* fused_float3;
};

#endif