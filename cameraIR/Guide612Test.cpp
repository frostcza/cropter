#include "Guide612.h"
#include <jetson-utils/cudaUtility.h>
#include <jetson-utils/glDisplay.h>
#include <jetson-utils/cudaColorspace.h>
#include <jetson-utils/imageIO.h>

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
    // int count = 0;
    // char filename[100];
    // 初始化
    GuideCamera::Init();

    glDisplay* dis = glDisplay::Create(NULL, 640, 512);

    // 全图测温 12~20ms
    // 此处Point2i.x对应width方向, Point2i.y对应height方向
    cv::Point2i pts[4] = { cv::Point2i(0,0), cv::Point2i(0, 511), cv::Point2i(639, 511), cv::Point2i(639, 0)};
    std::vector<float> temper;


    while(!signal_recieved)
    {
        GuideCamera::CaptureIRRGB(&dummy, &data, 1000);
        GuideCamera::CaptureIRY16(&dummy, &data_y16, 1000);
        dis->BeginRender();
        dis->RenderImage(data, 640, 512, IMAGE_RGB8, 0, 0);

        // count++;
        // if(count == 15)
        // {
        //     framenum++;
        //     sprintf(filename, "../../saved_image/IR/IR-%08d.jpg", framenum);
        //     saveImage(filename, data, GUIDE_CAM_W, GUIDE_CAM_H, IMAGE_RGB8, 90);
        //     count = 0;
        // }

        auto start = std::chrono::system_clock::now();
        temper = GuideCamera::calcTemper((short*) data_y16, pts);
        auto end = std::chrono::system_clock::now();
		int runtime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printf("%d ms\n", runtime);

        printf("[TEST] region max Temper:-->:%.1f℃    min Temper:-->:%.1f℃    mean Temper:-->:%.1f℃\n", temper[0], temper[1], temper[2]);
       
        // printf("[TEST] region max Temper:-->:%.1f℃    min Temper:-->:%.1f℃\n", temper[0], temper[1]);

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
