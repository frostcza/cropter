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

class ModuleDetection: public Yolov5
{
public:

    ModuleDetection(string engine_path, ModuleRIFT* moduleRIFT);
    ~ModuleDetection();

    /**
	 * @brief 执行检测的全过程, 将检测框, 物体温度和最热点位置画在img_vi和img_ir上
	 * @param img_vi 可见光图像
	 * @param img_ir 红外图像
	 * @param data_y16 Y16数据, 用于测温
	 * @param width 可见光图像w
     * @param height 可见光图像h
	 */
    void Detect(void* img_vi, void* img_ir, short* data_y16, int width, int height);

    /**
	 * @brief 对mRes中所有检测出的区域进行测温
	 * @param img_vi 可见光图像
	 * @param data_y16 Y16数据, 用于测温
	 * @param width 可见光图像w
     * @param height 可见光图像h
     * @param draw_box 为true时在红外图像上画出检测结果
	 */
    void getMaxTemper(void* img_ir, short* data_y16, int width, int height, bool draw_box);

    /**
	 * @brief 在可见光图像上画出检测结果
	 * @param img_vi 可见光图像
	 * @param width 可见光图像w
     * @param height 可见光图像h
	 */
    void drawBoxLabel(void* img_vi, int width, int height);
    vector<Detection> mRes; // 调用YOLOv5 doInference()后得到的检测结果

private:

    ModuleRIFT* mModuleRIFT;
    std::vector<float> dispTempers;
    cudaFont* mFont;
    bool calcTemperature;
    std::vector<cv::Point2i> hotspots;
    std::vector<std::string> names = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", 
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", 
        "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", 
        "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        };
};

#endif
