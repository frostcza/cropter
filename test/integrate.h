#ifndef __INTEGRATE_H__
#define __INTEGRATE_H__

#include "modules/ModuleCamera.h"
#include "modules/ModuleGPIO.h"
#include "modules/ModuleRIFT.h"
#include "modules/ModuleDetection.h"
#include "modules/ModuleFusion.h"
#include "modules/ModuleI2C.h"
#include "cameraIR/Guide612.h"
#include <jetson-utils/glDisplay.h>

struct saveThreadArgs
{
	void* frameIR;
	void* frameVIS;
	void* frameDetect;
	int framenum;
	int IRW;
	int IRH;
	int VIW;
	int VIH;
};

struct RunOption
{
    bool use_GPIO;
    bool shrink_picture;
};

class Integrate
{
public:
    Integrate(RunOption opt);
    ~Integrate();
    void mainLoop();

    ModuleGPIO* gpio;
    glDisplay* dis;

private:
    void MemoryAlloc();
    bool Init();
    void startSaveThread(saveThreadArgs save_args);

    RunOption mOption;

    ModuleCamera* cameraIR;
    LiteGstCamera* gstVIS;
    ModuleCamera* cameraVIS;
    ModuleRIFT* rift;
    ModuleDetection* det;
    ModuleFusion* fusion;
    ModuleI2C* i2c;

    const int cameraVIS_W = 1920;
    const int cameraVIS_H = 1080;
    const int cameraIR_W = GUIDE_CAM_W;
    const int cameraIR_H = GUIDE_CAM_H;

    const std::string detection_engine = "../../detection/yolov5n.engine";

    cv::Mat Homography;
	cv::Mat imIR;
	cv::Mat imVIS;
	cv::Mat imIRWarp;
    cv::Mat imFused;
	// cv::Mat imIRWarpSplit[3];
	// cv::Mat imVISSplit[3];

    void* frameIR;
	void* frameVIS;
    void* frameY16;

    void* frameVISSmall;
    void* frameIRWarp;

    void* frameFused;
    void* frameFusedLarge;

    void* frameVISDetected;
	void* frameVISDetectedSmall;

	void* frameIRLarge;

	void* frameIR_copy;
	void* frameVIS_copy;
	void* frameDetect_copy;

    char title[256];
    int count;
	int runtime;
	int disptime;
    int framenum;
    int dummy;
};

#endif