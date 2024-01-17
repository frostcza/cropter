#ifndef __MODULE_FUSION_H__
#define __MODULE_FUSION_H__

#include "fusion/IFCNN.h"
#include <string>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

enum Precision
{
    INT8 = 0, FP16 = 1
};

class ModuleFusion
{
public:

    ModuleFusion(Precision p);
    ~ModuleFusion();

    /**
	 * @brief 融合接口，要求图像尺寸相同，数据类型为RGB8
	 * @param ir 红外图像
	 * @param vi 可见光图像
	 * @param fused_image 返回的融合图像，数据类型为RGB8
	 */
    void Fuse(uchar3* ir, uchar3* vi, void* fused_image);

    IFCNN* ifcnn;
};

#endif
