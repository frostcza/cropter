#include <cstdint>
#include <iostream>
#include <unistd.h>
#include <signal.h>
#include <chrono>
#include <sys/time.h>
#include <fstream>

#include "modules/ModuleCamera.h"
#include "modules/ModuleI2C.h"
#include "cameraIR/Guide612.h"
#include <jetson-utils/cudaUtility.h>
#include <jetson-utils/cudaDraw.h>
#include <jetson-utils/glDisplay.h>

volatile bool signal_recieved = false;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		printf("received SIGINT\n");
		signal_recieved = true;
	}
}

void test()
{
	float actual_temp;
	printf("Now the blackbody's Temper is: ");
	std::cin >> actual_temp;
	// Camera
	if( !GuideCamera::Init())
	{
		printf("[Test] failed to initialize IR camera\n");
		return;
	}
	ModuleCamera* cameraIR = new ModuleCamera(NULL, CAM_IR);

	// I2C 
	ModuleI2C* i2c = new ModuleI2C();
	i2c->Start();

	// Display
    glDisplay* dis = glDisplay::Create(NULL, 640, 512);
	void* frameIR = NULL;
	void* frameY16 = NULL;

	// Time recording
	int count = 0;
	struct timeval start, end;
	gettimeofday(&start, NULL);

	// Other settings
	int dummy = 0;
	cv::Point2i pts[4] = { cv::Point2i(250,200), cv::Point2i(350, 200), cv::Point2i(350, 300), cv::Point2i(250, 300)};
    TemperResult temper;
    unsigned char cmd[] = {0x55, 0xAA, 0x07, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00, 0x87, 0xF0};

	// txt handle
	std::ofstream outfile("out_4_6.txt", std::ios::app);

	// The main cycle
	while(!signal_recieved)
	{
		GuideCamera::CaptureIRRGB(&dummy, &frameIR, 1000);
        GuideCamera::CaptureIRY16(&dummy, &frameY16, 1000);
        dis->BeginRender();
        dis->RenderImage(frameIR, 640, 512, IMAGE_RGB8, 0, 0);

        temper = GuideCamera::calcTemper((short*) frameY16, pts, HOTSPOT);
        // printf("[TEST] max Temper:-->:%.1f℃  mean Temper:-->:%.1f℃  section_avg Temper:-->:%.1f℃\n", temper.maxTemper, temper.meanTemper, temper.sectionAverage);
        cv::Point2i hotspot = temper.maxTemperLoc;

		// draw
        float4 marker_color = make_float4(255.0f, 0.0f, 0.0f, 255.0f);
        cudaDrawLine(frameIR, GUIDE_CAM_W, GUIDE_CAM_H, IMAGE_RGB8, hotspot.x - 3, hotspot.y, hotspot.x + 3, hotspot.y, marker_color);
        cudaDrawLine(frameIR, GUIDE_CAM_W, GUIDE_CAM_H, IMAGE_RGB8, hotspot.x, hotspot.y - 3, hotspot.x, hotspot.y + 3, marker_color);
        float4 color = make_float4(0.0f, 255.0f, 0.0f, 255.0f);
        cudaDrawLine(frameIR, GUIDE_CAM_W, GUIDE_CAM_H, IMAGE_RGB8, pts[0].x, pts[0].y, pts[1].x, pts[1].y, color);
        cudaDrawLine(frameIR, GUIDE_CAM_W, GUIDE_CAM_H, IMAGE_RGB8, pts[1].x, pts[1].y, pts[2].x, pts[2].y, color);
        cudaDrawLine(frameIR, GUIDE_CAM_W, GUIDE_CAM_H, IMAGE_RGB8, pts[2].x, pts[2].y, pts[3].x, pts[3].y, color);
        cudaDrawLine(frameIR, GUIDE_CAM_W, GUIDE_CAM_H, IMAGE_RGB8, pts[3].x, pts[3].y, pts[0].x, pts[0].y, color);

		// save
        count++;
        if(count == 300)
        {
			gettimeofday(&end, NULL);
			printf("Time elapse: %d\n", (int)(end.tv_sec - start.tv_sec));
            guide_usb_sendcommand(cmd, 12);
			printf("Actual Temp: %.2f℃\n", actual_temp);
			printf("Measured Temp: %.2f℃\n", temper.maxTemper);
			printf("Focal plane Temp: %.2f℃\n", GuideCamera::focalPlane);
            printf("Enviornment Temp: %.2f℃\n", i2c->mTemperature);
            printf("Enviornment Hum: %.2f%%\n", i2c->mHumidity);
			if(temper.maxTemper != 0.0 && GuideCamera::focalPlane != 0.0)
			{
				outfile << std::setw(4) << (int)(end.tv_sec - start.tv_sec) << "\t"
				<< std::fixed << std::setprecision(2) << actual_temp << "\t"
				<< std::fixed << std::setprecision(2) << temper.maxTemper << "\t"
				<< std::fixed << std::setprecision(2) << GuideCamera::focalPlane << "\t"
				<< std::fixed << std::setprecision(2) << i2c->mTemperature << "\t"
				<< std::fixed << std::setprecision(2) << i2c->mHumidity << std::endl;
			}
            count = 0;
        }

        GuideCamera::CaptureRGBFinish(frameIR);
        GuideCamera::CaptureY16Finish(frameY16);
        dis->EndRender();

        if(!dis->IsStreaming())
            signal_recieved = true;
		
	}

	// Stopping sub threads
	i2c->Stop();
	i2c->Join();

	// Release resources
	SAFE_DELETE(i2c);
	SAFE_DELETE(dis);
	GuideCamera::DeInit();
	outfile.close();

	printf("[Test] shut down\n");
}

int main(int argc, char** argv)
{
    // catch Ctrl + C signal
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("can't catch SIGINT\n");
	
	test();
	return 0;
}
