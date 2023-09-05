#include "ModuleDetection.h"
#include <jetson-utils/cudaDraw.h>
#include <chrono>

ModuleDetection::ModuleDetection(string engine_path, ModuleRIFT* moduleRIFT):Yolov5(engine_path)
{
    mModuleRIFT = moduleRIFT;
    mFont = cudaFont::Create(40);
    calcTemperature = true;
    printf("[Detection] detection engine initialize done \n");
}

ModuleDetection::~ModuleDetection()
{
    SAFE_DELETE(mFont);
}

void ModuleDetection::Detect(void* img_vi, void* img_ir, short* data_y16, int width, int height)
{
    doInference(img_vi, width, height, mRes); // 10~40ms
    
    // auto start = std::chrono::system_clock::now();
    if(calcTemperature)
    {
        getMaxTemper(img_ir, data_y16, width, height, false); // 3~4 ms per object
    }
    // auto end = std::chrono::system_clock::now();
    // int runtime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // printf("[Detection] getMaxTemper use %d ms \n", runtime);
    drawBoxLabel(img_vi, width, height); // 0ms
}

void ModuleDetection::getMaxTemper(void* img_ir, short* data_y16, int width, int height, bool draw_box)
{
    cv::Mat Homography = mModuleRIFT->getTransMat();
    cv::Mat H_tanspose;
    cv::invert(Homography, H_tanspose, cv::DECOMP_SVD);
    dispTempers.clear();
    hotspots.clear();

    for (size_t i = 0; i < mRes.size(); i++)
    {
        cv::Point2i pts[4];
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
            pts[j].x = (int)round(pts[j].x * (float)GUIDE_CAM_W / (float)width);
            pts[j].y = (int)round(pts[j].y * (float)GUIDE_CAM_H / (float)height);
            float x_trans = H_tanspose.ptr<float>(0)[0] * pts[j].x + H_tanspose.ptr<float>(0)[1] * pts[j].y + H_tanspose.ptr<float>(0)[2];
            float y_trans = H_tanspose.ptr<float>(1)[0] * pts[j].x + H_tanspose.ptr<float>(1)[1] * pts[j].y + H_tanspose.ptr<float>(1)[2];
            float scale = H_tanspose.ptr<float>(2)[0] * pts[j].x + H_tanspose.ptr<float>(2)[1] * pts[j].y + H_tanspose.ptr<float>(2)[2];
            x_trans /= scale;
            y_trans /= scale;
            pts[j].x = (int)round(x_trans);
            pts[j].y = (int)round(y_trans);
            if(pts[j].x < 0) pts[j].x = 0;
            if(pts[j].x > GUIDE_CAM_W - 1) pts[j].x = GUIDE_CAM_W - 1;
            if(pts[j].y < 0) pts[j].y = 0;
            if(pts[j].y > GUIDE_CAM_H - 1) pts[j].y = GUIDE_CAM_H - 1;
        }

        if(std::abs(pts[0].x - pts[2].x) < 10  ||  std::abs(pts[0].y - pts[2].y) < 10
                || std::abs(pts[1].x - pts[3].x) < 10  ||  std::abs(pts[1].y - pts[3].y) < 10)
        {
            dispTempers.push_back(0.0);
            hotspots.push_back(cv::Point2i(0,0));
            continue;
        }
        
        TemperResult temper_res = GuideCamera::calcTemper(data_y16, pts, HOTSPOT); // 3ms
        dispTempers.push_back(temper_res.maxTemper);
        // if(temper_res.sectionAverage > 30.0)
        //     dispTempers.push_back(temper_res.maxTemper);
        // else
        //     dispTempers.push_back(temper_res.meanTemper);

        cv::Point2i hotspot = temper_res.maxTemperLoc;
        if(draw_box)
        {
            float4 color = make_float4(0.0f, 255.0f, 0.0f, 255.0f);
            float4 marker_color = make_float4(255.0f, 0.0f, 0.0f, 255.0f);
            cudaDrawLine(img_ir, GUIDE_CAM_W, GUIDE_CAM_H, IMAGE_RGB8, pts[0].x, pts[0].y, pts[1].x, pts[1].y, color);
            cudaDrawLine(img_ir, GUIDE_CAM_W, GUIDE_CAM_H, IMAGE_RGB8, pts[1].x, pts[1].y, pts[2].x, pts[2].y, color);
            cudaDrawLine(img_ir, GUIDE_CAM_W, GUIDE_CAM_H, IMAGE_RGB8, pts[2].x, pts[2].y, pts[3].x, pts[3].y, color);
            cudaDrawLine(img_ir, GUIDE_CAM_W, GUIDE_CAM_H, IMAGE_RGB8, pts[3].x, pts[3].y, pts[0].x, pts[0].y, color);

            cudaDrawLine(img_ir, GUIDE_CAM_W, GUIDE_CAM_H, IMAGE_RGB8, hotspot.x - 8, hotspot.y, hotspot.x + 8, hotspot.y, marker_color, 3);
            cudaDrawLine(img_ir, GUIDE_CAM_W, GUIDE_CAM_H, IMAGE_RGB8, hotspot.x, hotspot.y - 8, hotspot.x, hotspot.y + 8, marker_color, 3);
        }

        float hotspot_trans_x = Homography.ptr<float>(0)[0] * hotspot.x + Homography.ptr<float>(0)[1] * hotspot.y + Homography.ptr<float>(0)[2];
        float hotspot_trans_y = Homography.ptr<float>(1)[0] * hotspot.x + Homography.ptr<float>(1)[1] * hotspot.y + Homography.ptr<float>(1)[2];
        float hotspot_trans_scale = Homography.ptr<float>(2)[0] * hotspot.x + Homography.ptr<float>(2)[1] * hotspot.y + Homography.ptr<float>(2)[2];
        hotspot_trans_x /= hotspot_trans_scale;
        hotspot_trans_y /= hotspot_trans_scale;
        hotspot.x = (int)round(hotspot_trans_x * (float)width / (float)GUIDE_CAM_W);
        hotspot.y = (int)round(hotspot_trans_y * (float)height / (float)GUIDE_CAM_H);
        if(hotspot.x > width - 9) hotspot.x = width - 9;
        if(hotspot.x < 8) hotspot.x = 8;
        if(hotspot.y > height - 9) hotspot.y = height - 9;
        if(hotspot.y < 8) hotspot.y = 8;
        hotspots.push_back(hotspot);
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

        string name = names[(int)mRes[i].class_id];
        if(calcTemperature && dispTempers[i] != 0.0)
        {
            float4 marker_color = make_float4(255.0f, 0.0f, 0.0f, 255.0f);
            cudaDrawLine(img_vi, width, height, IMAGE_RGB8, hotspots[i].x - 8, hotspots[i].y, hotspots[i].x + 8, hotspots[i].y, marker_color, 3);
            cudaDrawLine(img_vi, width, height, IMAGE_RGB8, hotspots[i].x, hotspots[i].y - 8, hotspots[i].x, hotspots[i].y + 8, marker_color, 3);
            char temper[10];
            snprintf(temper, 6, "%.1f", dispTempers[i]);
            string c = temper;
            labels.push_back(std::pair<std::string, int2>(name+" "+c, {x+5,y+5}));
        }
        else
        {
            labels.push_back(std::pair<std::string, int2>(name, {x+5,y+5}));
        }

    }

    mFont->OverlayText(img_vi, IMAGE_RGB8, width, height, labels, make_float4(255,0,0,255));
}
