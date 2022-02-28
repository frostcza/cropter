#ifndef __MODULE_DETECTION_H__
#define __MODULE_DETECTION_H__

#include "detection/yolov5.h"
#include <jetson-utils/cudaFont.h>
#include <string>
#include <vector>
#include "modules/ModuleRIFT.h"
#include "cameraIR/Guide612.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

class ModuleDetection:Yolov5
{
public:

    ModuleDetection(string engine_path, ModuleRIFT* moduleRIFT);
    ~ModuleDetection();
    void Detect(void* img_vi, short* data_y16, int width, int height);


private:

    void getMaxTemper(void* img_vi, short* data_y16, int width, int height);
    void drawBoxLabel(void* img_vi, int width, int height);

    ModuleRIFT* mModuleRIFT;
    vector<Detection> mRes;
    std::vector<float> maxTempers;
    cudaFont* mFont;
    bool calcTemperature;
};

#endif
