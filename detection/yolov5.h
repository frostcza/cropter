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
using namespace Yolo; // See namespace Yolo in yololayer.h, it defined INPUT_H INPUT_W and struct Detetion

class Yolov5
{
public:

    Yolov5(string engine_path);
    ~Yolov5();

    //! \brief Detect objects using YOLOv5n
    //! \param img_to_detect A mapped memory buffer
    //! \param width The width of the original image
    //! \param height The height of the original image
    //! \param det_result The detection result. See yololayer.h struct Detection.
    //!
    //! \return True if the detection were execute successfully.
    bool doInference(void* img_to_detect, int width, int height, std::vector<Detection> &det_result);

    string engine_name;

private:

    bool Init();

    /**
	 * @brief bbox[4] stores center.x, center.y of the box, and image.w, image.h based on 640*640 size. 
     * We should convert it to lefttop.x, lefttop.y, image.w, image.h on the orginal image size. 
	 * @param width The width of the original image
	 * @param height The height of the original image
	 * @param bbox Yolo::Detection.bbox 
	 */
    void bbox2rect(int width, int height, float bbox[4]);

    // Calculate the IOU of lbox and rbox
    float iou(float lbox[4], float rbox[4]);

    // Custom compare function for std::sort
    static inline bool cmp(const Detection& a, const Detection& b) { return a.conf > b.conf;};

    /**
	 * @brief Non-Maximum Suppression based on IOU
	 * @param network_output The raw inference output
	 * @param res The output
	 * @param conf_thresh Detection that has confidence < conf_thresh will be reject
     * @param nms_thresh IOU thresh 
	 */
    void nms(float *network_output, std::vector<Detection>& res, float conf_thresh, float nms_thresh);

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
    
    float* buffers[2]; // buffer[0] is the network input, i.e, 640*640 image
    float prob[OUTPUT_SIZE]; // buffer[1] is the network output, i.e, vector<Detection>
    void* resized;
};


#endif
