#include <cstdint>
#include <iostream>
#include <vector>
#include <unistd.h>
#include <signal.h>
#include <chrono>

#include "modules/ModuleCamera.h"
#include "modules/ModuleGPIO.h"
#include "modules/ModuleRIFT.h"
#include "modules/ModuleDetection.h"
#include "cameraIR/Guide612.h"
#include <jetson-utils/cudaUtility.h>
#include <jetson-utils/glDisplay.h>
#include <jetson-utils/cudaResize.h>

volatile bool signal_recieved = false;
mode keyboard_mode = Fuse;

// 检测ctrl+C信号,SIGINT信号即ctrl+C信号.
void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		printf("received SIGINT\n");
		signal_recieved = true;
	}
}

void sig_handler_mode(int signo)
{
	if( signo == SIGTSTP )
	{
		printf("mode switch\n");
		if (keyboard_mode == IR)
			keyboard_mode = VI;
		else if (keyboard_mode == VI)
			keyboard_mode = Fuse;
		else if (keyboard_mode == Fuse)
			keyboard_mode = IR;
	}
}

void test()
{
	if( !GuideCamera::Init())
	{
		printf("[Test] failed to initialize IR camera\n");
		return;
	}
	ModuleCamera* cameraIR = ModuleCamera::Create(NULL, CAM_IR, 8);

    LiteGstCamera* gstVIS = LiteGstCamera::Create(1920, 1080, "/dev/video0");
    if( !gstVIS )
	{
		printf("[Test] failed to initialize VIS camera\n");
		return;
	}
	ModuleCamera* cameraVIS = ModuleCamera::Create(gstVIS, CAM_VIS, 8);

    ModuleGPIO* gpio = new ModuleGPIO(cameraIR, cameraVIS);

	ModuleRIFT* rift = new ModuleRIFT(cameraIR, cameraVIS);
	cv::Mat Homography;
	cv::Mat imIR;
	cv::Mat imVIS;
	cv::Mat fused;
	void* frameFused = NULL;
	cudaAllocMapped(&frameFused, BYTES_RGB);
	void* frameFusedResized = NULL;
	cudaAllocMapped(&frameFusedResized, 1280*1024*sizeof(uchar3));

	std::string inference_engine = "../../detection/yolov5n.engine"; 
	ModuleDetection* det = new ModuleDetection(inference_engine, rift);

	cameraIR->Start();
	cameraVIS->Start();
	// gpio->Start();
	rift->Start();

    glDisplay* dis = glDisplay::Create(NULL, 1920, 1080);
	// glDisplay* dis = glDisplay::Create(NULL, 640, 512);
	void* frameIR = NULL;
	void* frameVIS = NULL;
	void* frameY16 = NULL;
	void* frameIRResized = NULL;
	cudaAllocMapped(&frameIRResized, 1280*1024*sizeof(uchar3));
	char str[256];

	int dummy = 0;

	// gpio->mode_result = Fuse;
	int count = 0;
	int runtime = 0;
	int disptime = 0;

	while(!signal_recieved)
	{
		auto start = std::chrono::system_clock::now();
		gpio->mode_result = keyboard_mode;
        switch(gpio->mode_result)
		{
			case IR:
			{
				// if(!cameraIR->Read(&dummy, (void**)&frameIR, UINT64_MAX))
				// {
				// 	printf("[Test] Cannot get IR data\n");
				// }
				if(!GuideCamera::CaptureIRRGB(&dummy, &frameIR, UINT64_MAX))
		    	{
					printf("[Test] Cannot get IR data\n");
				}
				cudaResize((uchar3*)frameIR, cameraIR->GetWidth(), cameraIR->GetHeight(), (uchar3*)frameIRResized, 1280, 1024);
				dis->BeginRender();
				// dis->RenderImage(frameIR, GUIDE_CAM_W, GUIDE_CAM_H, IMAGE_RGB8, 0, 0);
				dis->RenderImage(frameIRResized, 1280, 1024, IMAGE_RGB8, 0, 0);
				dis->EndRender();
				// cameraIR->ReadFinish(frameIR);
				GuideCamera::CaptureRGBFinish(frameIR);
				break;
			}
			case VI:
			{
				if(!cameraVIS->Read(&dummy, (void**)&frameVIS, UINT64_MAX))
		    	{
					printf("[Test] Cannot get VIS data\n");
				}
				if(!GuideCamera::CaptureIRY16(&dummy, &frameY16, UINT64_MAX))
		    	{
					printf("[Test] Cannot get Y16 data\n");
				}
				det->Detect(frameVIS, (short*)frameY16, cameraVIS->GetWidth(), cameraVIS->GetHeight());
				dis->BeginRender();
				dis->RenderImage(frameVIS, cameraVIS->GetWidth(), cameraVIS->GetHeight(), IMAGE_RGB8, 0, 0);
				dis->EndRender();
				cameraVIS->ReadFinish(frameVIS);
				GuideCamera::CaptureY16Finish(frameY16);
				break;
			}
			case Fuse:
			{
				if(!cameraIR->Read(&dummy, (void**)&frameIR, UINT64_MAX))
				{
					printf("[Test] Cannot get IR data\n");
				}
				CUDA(cudaDeviceSynchronize());
        		imIR = cv::Mat(cameraIR->GetHeight(), cameraIR->GetWidth(), CV_8UC3, frameIR);
				cameraIR->ReadFinish(frameIR);

				if(!cameraVIS->Read(&dummy, (void**)&frameVIS, UINT64_MAX))
		    	{
					printf("[Test] Cannot get VIS data\n");
				}
				imVIS = cv::Mat(cameraVIS->GetHeight(), cameraVIS->GetWidth(), CV_8UC3, frameVIS);
				cameraVIS->ReadFinish(frameVIS);

				cv::resize(imVIS, imVIS, imIR.size());
				Homography = rift->getTransMat();
				
				cv::warpPerspective(imIR, fused, Homography, imIR.size());
				cv::addWeighted(imVIS, 0.5, fused, 0.5, 0.0, fused);
				
				cudaMemcpy(frameFused, fused.data, BYTES_RGB, cudaMemcpyHostToDevice);
				cudaResize((uchar3*)frameFused, cameraIR->GetWidth(), cameraIR->GetHeight(), (uchar3*)frameFusedResized, 1280, 1024);
				dis->BeginRender();
				// dis->RenderImage(frameFused, GUIDE_CAM_W, GUIDE_CAM_H, IMAGE_RGB8, 0, 0);
				dis->RenderImage(frameFusedResized, 1280, 1024, IMAGE_RGB8, 0, 0);
				dis->EndRender();

				break;

			}
			case Unknown:
			{
				printf("[Test] Unknown mode\n");
				break;
			}
			default:
				break;
		};

		count++;
		auto end = std::chrono::system_clock::now();
		runtime += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		if(count == 10)
		{
			disptime = runtime / 10;
			runtime = 0;
			count = 0;
		}

        // sprintf(str, "Camera Viewer | %.0f FPS",  dis->GetFPS());
		sprintf(str, "Camera Viewer | %.0f FPS",  1000.0 / float(disptime));
        dis->SetTitle(str);	
        if( dis->IsClosed() )
            signal_recieved = true;


	}

	cameraIR->Stop();
	cameraVIS->Stop();
	// gpio->Stop();
	rift->Stop();

    cameraIR->Join();
	cameraVIS->Join();
	// gpio->Join();
	rift->Join();

	SAFE_DELETE(cameraIR);
	SAFE_DELETE(cameraVIS);
	SAFE_DELETE(gpio);
    SAFE_DELETE(gstVIS);
	SAFE_DELETE(rift)
	SAFE_DELETE(det)
	SAFE_DELETE(dis);
	GuideCamera::DeInit();

	CUDA_FREE_HOST(frameIRResized);
	CUDA_FREE_HOST(frameFused);
	CUDA_FREE_HOST(frameFusedResized);
	printf("[Test] shut down\n");
}

int main(int argc, char** argv)
{
    // catch Ctrl + C signal
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("can't catch SIGINT\n");

	// catch Ctrl + Z signal
	if( signal(SIGTSTP, sig_handler_mode) == SIG_ERR)
		printf("can't catch SIGTSTP");
	
	test();
	return 0;
}
