#include "yolov5.h"
#include "myCudaResize.h"
#include <jetson-utils/imageIO.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
{\
    cudaError_t error_code = callstr;\
    if (error_code != cudaSuccess) {\
        std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
    }\
}
#endif  // CUDA_CHECK


void Yolov5::bbox2rect(int width, int height, float bbox[4]) 
{
    float l, r, t, b;
    float r_w = INPUT_W / (width * 1.0);
    float r_h = INPUT_H / (height * 1.0);
    if (r_h > r_w) 
    {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (INPUT_H - r_w * height) / 2;
        b = bbox[1] + bbox[3] / 2.f - (INPUT_H - r_w * height) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } 
    else 
    {
        l = bbox[0] - bbox[2] / 2.f - (INPUT_W - r_h * width) / 2;
        r = bbox[0] + bbox[2] / 2.f - (INPUT_W - r_h * width) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    bbox[0] = round(l);
    bbox[1] = round(t);
    bbox[2] = round(r - l);
    bbox[3] = round(b - t);
}

float Yolov5::iou(float lbox[4], float rbox[4]) 
{
    float interBox[] = 
    {
        (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
        (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
        (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
        (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

void Yolov5::nms(std::vector<Detection>& res, float *output, float conf_thresh, float nms_thresh) 
{
    int det_size = sizeof(Detection) / sizeof(float);
    std::map<float, std::vector<Detection>> m;
    for (int i = 0; i < output[0] && i < MAX_OUTPUT_BBOX_COUNT; i++)
    {
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) 
    {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) 
        {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) 
            {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) 
                {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}


Yolov5::Yolov5(string engine_path)
{
    engine_name = engine_path;
    if(!Init())
    {
        printf("[Yolov5] Init failed.\n");
    }
}

Yolov5::~Yolov5()
{
    // context->destroy();
    // engine->destroy();
    // runtime->destroy();
    SAFE_DELETE(context);
    SAFE_DELETE(engine);
    SAFE_DELETE(runtime);

    cudaFreeHost(buffers[0]);
    cudaFreeHost(buffers[1]);
    cudaFreeHost(resized);

    // cudaStreamDestroy(stream);
}


bool Yolov5::Init()
{
    printf("[Yolov5] Loading detection engine ...\n");
    // initLibNvInferPlugins(&gLogger, "");
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) 
    {
        printf("[Yolov5] Read engine file failed.\n");
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
        printf("[Yolov5] Create runtime falied.\n");
        return false;
    }

    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    if(!engine)
    {        
        printf("[Yolov5] Deserialize engine falied.\n");
        return false;
    }
    context = engine->createExecutionContext();
    if(!context)
    {
        printf("[Yolov5] Create context failed.\n");
        return false;
    }
    delete[] trtModelStream;

    if(engine->getBindingIndex(INPUT_BLOB_NAME) != 0 || engine->getBindingIndex(OUTPUT_BLOB_NAME) != 1)
    {
        printf("[Yolov5] Get binding index failed.\n");
        return false;
    }

    cudaAllocMapped((void**)&buffers[0], INPUT_H * INPUT_W * sizeof(float3));
    cudaAllocMapped((void**)&buffers[1], OUTPUT_SIZE * sizeof(float));
    cudaAllocMapped((void**)&resized, INPUT_H * INPUT_W * sizeof(float3));

    // CUDA_CHECK(cudaStreamCreate(&stream));

    return true;
}

bool Yolov5::doInference(void* img_to_detect, int width, int height, std::vector<Detection> &det_result)
{
    det_result.clear();
    cudaResizeNoStretch((uchar3*) img_to_detect, width, height, (float3*) resized, INPUT_W, INPUT_H);
    // saveImage("temp.jpg", resized, INPUT_W, INPUT_H, IMAGE_RGB32F);
    cudaPacked2Planner((float3*) resized, INPUT_W, INPUT_H, (float*)buffers[0]);

    context->execute(1, (void**)buffers);
    CUDA_CHECK(cudaMemcpy(prob, buffers[1], OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    // context->enqueue(1, (void**)buffers, stream, nullptr);
    // CUDA_CHECK(cudaMemcpyAsync(prob, buffers[1], OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    // cudaStreamSynchronize(stream);

    nms(det_result, prob, CONF_THRESH, NMS_THRESH);
    // printf("[yolov5] detections before nms: %d, after nms: %ld.\n", (int)prob[0], det_result.size());

    for (size_t j = 0; j < det_result.size(); j++)
    {
        bbox2rect(width, height, det_result[j].bbox);
    }
    return true;

    // 外部调用cudaDrawLine来画框，font->OverlayText来写出label，这里不做处理

}

