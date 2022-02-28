#include "ModuleDetection.h"
#include <jetson-utils/cudaDraw.h>

ModuleDetection::ModuleDetection(string engine_path, ModuleRIFT* moduleRIFT):Yolov5(engine_path)
{
    mModuleRIFT = moduleRIFT;
    mFont = cudaFont::Create(30);
    calcTemperature = false;
    printf("[Detection] detection engine initialize done \n");
}

ModuleDetection::~ModuleDetection()
{
    SAFE_DELETE(mFont);
}

void ModuleDetection::Detect(void* img_vi, short* data_y16, int width, int height)
{
    doInference(img_vi, width, height, mRes);
    if(calcTemperature)
    {
        getMaxTemper(img_vi, data_y16, width, height);
    }
    drawBoxLabel(img_vi, width, height);
}

void ModuleDetection::getMaxTemper(void* img_vi, short* data_y16, int width, int height)
{
    cv::Mat Homography = mModuleRIFT->getTransMat();
    cv::Mat H_tanspose;
    cv::invert(Homography, H_tanspose, cv::DECOMP_SVD);
    maxTempers.clear();
    for (size_t i = 0; i < mRes.size(); i++)
    {   
        std::vector<cv::Point2i> pts(4);
        int x = (int)mRes[i].bbox[0];
        int y = (int)mRes[i].bbox[1];
        int w = (int)mRes[i].bbox[2];
        int h = (int)mRes[i].bbox[3];
        pts[0] = cv::Point2i(x, y);
        pts[1] = cv::Point2i(x + w, y);
        pts[2] = cv::Point2i(x + w, y + h);
        pts[3] = cv::Point2i(x, y + h);
        
        for(int j = 0; j < 4; j++)
        {
            float x_trans = H_tanspose.ptr<float>(0)[0] * pts[j].x + H_tanspose.ptr<float>(0)[1] * pts[j].y + H_tanspose.ptr<float>(0)[2];
            float y_trans = H_tanspose.ptr<float>(1)[0] * pts[j].x + H_tanspose.ptr<float>(1)[1] * pts[j].y + H_tanspose.ptr<float>(1)[2];
            float scale = H_tanspose.ptr<float>(2)[0] * pts[j].x + H_tanspose.ptr<float>(2)[1] * pts[j].y + H_tanspose.ptr<float>(2)[2];
            x_trans /= scale;
            y_trans /= scale;
            x_trans = x_trans > width - 1 ? height - 1 : x_trans;
            x_trans = x_trans < 0 ? 0 : x_trans;
            y_trans = y_trans > width - 1 ? height - 1 : y_trans;
            y_trans = y_trans < 0 ? 0 : y_trans;
            pts[j].x = round(x_trans * (float)GUIDE_CAM_W / (float)width);
            pts[j].y = round(y_trans * (float)GUIDE_CAM_H / (float)height);
        }

        std::vector<float> temper = GuideCamera::calcTemper(data_y16, pts.data());

        maxTempers.push_back(temper[0]);

        #if 1
        float4 color = make_float4(0.0f, 255.0f, 0.0f, 255.0f);
        cudaDrawLine(img_vi, width, height, IMAGE_RGB8, pts[0].x, pts[0].y, pts[1].x, pts[1].y, color);
        cudaDrawLine(img_vi, width, height, IMAGE_RGB8, pts[1].x, pts[1].y, pts[2].x, pts[2].y, color);
        cudaDrawLine(img_vi, width, height, IMAGE_RGB8, pts[2].x, pts[2].y, pts[3].x, pts[3].y, color);
        cudaDrawLine(img_vi, width, height, IMAGE_RGB8, pts[3].x, pts[3].y, pts[0].x, pts[0].y, color);
        #endif
    }
}

void ModuleDetection::drawBoxLabel(void* img_vi, int width, int height)
{
    int x, y, w, h;
    std::vector< std::pair< std::string, int2 > > labels;
    for (size_t i = 0; i < mRes.size(); i++)
    {
        x = (int)mRes[i].bbox[0];
        y = (int)mRes[i].bbox[1];
        w = (int)mRes[i].bbox[2];
        h = (int)mRes[i].bbox[3];
        float4 color = make_float4(0.0f, 0.0f, 255.0f, 255.0f);
        cudaDrawLine(img_vi, width, height, IMAGE_RGB8, x, y, x+w, y, color);
        cudaDrawLine(img_vi, width, height, IMAGE_RGB8, x, y, x, y+h, color);
        cudaDrawLine(img_vi, width, height, IMAGE_RGB8, x+w, y, x+w, y+h, color);
        cudaDrawLine(img_vi, width, height, IMAGE_RGB8, x, y+h, x+w, y+h, color);

        string a = "class ";
        string b = to_string((int)mRes[i].class_id);
        if(calcTemperature && maxTempers[i] != 0.0)
        {
            char* temper =  new(std::nothrow)char[10];
            snprintf(temper, 6, "%.1f", maxTempers[i]);
            string c(temper);
            delete []temper;
            labels.push_back(std::pair<std::string, int2>(a+b+" T="+c, {x+5,y+5}));
        }
        else
        {
            labels.push_back(std::pair<std::string, int2>(a+b, {x+5,y+5}));
        }

    }

    mFont->OverlayText(img_vi, IMAGE_RGB8, width, height, labels, make_float4(255,0,0,255));
}
