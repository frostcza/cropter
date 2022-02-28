#ifndef __YOLOV5_H__
#define __YOLOV5_H__

#include "yololayer.h"
#include <iostream>
#include <fstream>
#include <string>
#include <jetson-utils/cudaUtility.h>
#include <jetson-utils/cudaMappedMemory.h>
#include <map>
#include <algorithm>
#include <math.h>

using namespace nvinfer1;
using namespace std;
// See namespace Yolo in yololayer.h, it defined INPUT_H INPUT_W and struct Detetion
using namespace Yolo;

class Yolov5
{
public:

    Yolov5(string engine_path);
    ~Yolov5();

    //! \param img_to_detect A mapped memory buffer
    //! \param width The width of the original image
    //! \param height The height of the original image
    //! \param boxes The detection result
    //!
    //! \return True if the detection were execute successfully.
    bool doInference(void* img_to_detect, int width, int height, std::vector<Detection> &det_result);

    string engine_name;

private:

    bool Init();
    void bbox2rect(int width, int height, float bbox[4]);
    float iou(float lbox[4], float rbox[4]);
    static inline bool cmp(const Detection& a, const Detection& b) { return a.conf > b.conf;};
    void nms(std::vector<Detection>& res, float *output, float conf_thresh, float nms_thresh = 0.5);


    // To create a builder, we first need to instantiate the ILogger interface
    class Logger : public ILogger           
    {
        void log(Severity severity, AsciiChar const * msg) noexcept override
        {
            // suppress info-level messages
            if (severity <= Severity::kWARNING)
                printf("[Detection] %s\n", msg);
        }
    } gLogger;
    
    // Parameters of Yolov5 network
    const float CONF_THRESH = 0.5;
    const float NMS_THRESH = 0.4;
    static const int OUTPUT_SIZE = MAX_OUTPUT_BBOX_COUNT * sizeof(Detection) / sizeof(float) + 1;

    // Names of input and output, fixed when engine is created
    const char* INPUT_BLOB_NAME = "data";
    const char* OUTPUT_BLOB_NAME = "prob";

    // Nvinfer components
    // cudaStream_t stream;
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    
    // buffer[0] is the network input, i.e, 640*640 image
    // buffer[1] is the network output, i.e, vector<Detection>
    float* buffers[2];
    float prob[OUTPUT_SIZE];
    void* resized;


};


#endif
