#include "Guide612.h"
#include <jetson-utils/cudaUtility.h>
#include <jetson-utils/glDisplay.h>
#include <jetson-utils/cudaColorspace.h>
#include <jetson-utils/imageIO.h>
#include <jetson-utils/cudaDraw.h>

#include <stdio.h>
#include <signal.h>
#include <algorithm>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

bool signal_recieved = false;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		printf("received SIGINT\n");
		signal_recieved = true;
	}
}


int main()
{
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("can't catch SIGINT\n");

    // 用作身份记录
    int dummy = 0;

    // 图像数据
    void* data = NULL;
    void* data_y16 = NULL;

    // int framenum = 0;
    int count = 0;
    // char filename[100];
    // 初始化
    GuideCamera::Init();

    glDisplay* dis = glDisplay::Create(NULL, 640, 512);

    // 此处Point2i.x对应width方向, Point2i.y对应height方向
    cv::Point2i pts[4] = { cv::Point2i(0,0), cv::Point2i(0, 511), cv::Point2i(639, 511), cv::Point2i(639, 0)};
    TemperResult temper;

    unsigned char cmd[] = {0x55, 0xAA, 0x07, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00, 0x87, 0xF0};

    while(!signal_recieved)
    {
        GuideCamera::CaptureIRRGB(&dummy, &data, 1000);
        GuideCamera::CaptureIRY16(&dummy, &data_y16, 1000);
        dis->BeginRender();
        dis->RenderImage(data, 640, 512, IMAGE_RGB8, 0, 0);

        count++;
        if(count == 30)
        {
            guide_usb_sendcommand(cmd, 12);
            // framenum++;
            // sprintf(filename, "../../saved_image/IR/IR-%08d.jpg", framenum);
            // saveImage(filename, data, GUIDE_CAM_W, GUIDE_CAM_H, IMAGE_RGB8, 90);
            count = 0;
        }

        auto start = std::chrono::system_clock::now();
        temper = GuideCamera::calcTemper((short*) data_y16, pts, HOTSPOT);
        auto end = std::chrono::system_clock::now();
		int runtime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        // printf("%d ms\n", runtime);
        // printf("[TEST] max Temper:-->:%.1f℃  mean Temper:-->:%.1f℃  section_avg Temper:-->:%.1f℃\n", temper.maxTemper, temper.meanTemper, temper.sectionAverage);
        cv::Point2i hotspot = temper.maxTemperLoc;
        float4 marker_color = make_float4(255.0f, 0.0f, 0.0f, 255.0f);
        cudaDrawLine(data, GUIDE_CAM_W, GUIDE_CAM_H, IMAGE_RGB8, hotspot.x - 3, hotspot.y, hotspot.x + 3, hotspot.y, marker_color);
        cudaDrawLine(data, GUIDE_CAM_W, GUIDE_CAM_H, IMAGE_RGB8, hotspot.x, hotspot.y - 3, hotspot.x, hotspot.y + 3, marker_color);

        GuideCamera::CaptureRGBFinish(data);
        GuideCamera::CaptureY16Finish(data_y16);
        dis->EndRender();

        char str[256];
        sprintf(str, "Camera Viewer | %.0f FPS", dis->GetFPS());
        dis->SetTitle(str);	

        if(!dis->IsStreaming())
            signal_recieved = true;

    }


    printf("Exit!!!\n");
    SAFE_DELETE(dis);
    // 释放资源
    GuideCamera::DeInit();
    return 0;
}
