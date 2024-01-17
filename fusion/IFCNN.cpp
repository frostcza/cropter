#include "IFCNN.h"
#include <jetson-utils/cudaResize.h>
#include <jetson-utils/cudaColorspace.h>
#include "myCudaConvert.h"

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
{\
    cudaError_t error_code = callstr;\
    if (error_code != cudaSuccess) {\
        std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
    }\
}
#endif  // CUDA_CHECK


IFCNN::IFCNN(string engine_path) : engine_name(engine_path)
{
    if(!Init()) 
    {
        printf("[IFCNN] Init failed.\n");
    }
}

IFCNN::~IFCNN()
{
    SAFE_DELETE(context);
    SAFE_DELETE(engine);
    SAFE_DELETE(runtime);
    cudaStreamDestroy(stream);

    cudaFreeHost(buffers[0]);
    cudaFreeHost(buffers[1]);
    cudaFreeHost(buffers[2]);
    cudaFreeHost(fused_float3);
}

bool IFCNN::Init()
{
    printf("[IFCNN] Loading image fusion engine ...\n");
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) 
    {
        printf("[IFCNN] Read engine file failed.\n");
        return false;
    }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    file.read(trtModelStream, size);
    file.close();

    runtime = createInferRuntime(gLogger);
    if(!runtime)
    {
        printf("[IFCNN] Create runtime falied.\n");
        return false;
    }
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    if(!engine)
    {
        printf("[IFCNN] Deserialize engine falied.\n");
        return false;
    }
    context = engine->createExecutionContext();
    if(!context)
    {
        printf("[IFCNN] Create context falied.\n");
        return false;
    }
    delete[] trtModelStream;

    cudaStreamCreate(&stream);

    if(engine->getNbBindings() != 3 || engine->getBindingIndex(INPUT_IR_NAME) != 0 || 
        engine->getBindingIndex(INPUT_VI_NAME) != 1 || engine->getBindingIndex(OUTPUT_FUSED_NAME) != 2)
    {
        printf("[IFCNN] Get binding index failed.\n");
        return false;
    }

    cudaMalloc((void**)&buffers[0], INPUT_W * INPUT_H * sizeof(float3));
    cudaMalloc((void**)&buffers[1], INPUT_W * INPUT_H * sizeof(float3));
    cudaMalloc((void**)&buffers[2], INPUT_W * INPUT_H * sizeof(float3));
    cudaMalloc((void**)&fused_float3, INPUT_W * INPUT_H * sizeof(float3));

    return true;
}

void IFCNN::doInference(uchar3* ir, uchar3* vi, void* fused_image)
{
    cudaPackedToPlanner(ir, INPUT_W, INPUT_H, (float*)buffers[0]);
    cudaPackedToPlanner(vi, INPUT_W, INPUT_H, (float*)buffers[1]);

    context->enqueue(1, (void**)buffers, stream, nullptr);
    cudaStreamSynchronize(stream);

    cudaPlannerToPacked((float*)buffers[2], INPUT_W, INPUT_H, (float3*) fused_float3);
    cudaConvertColor(fused_float3, IMAGE_RGB32F, fused_image, IMAGE_RGB8, INPUT_W, INPUT_H, make_float2(0.0f, 1.0f));
}