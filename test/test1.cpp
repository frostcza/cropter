#include <cstdint>
#include <iostream>
#include <vector>
#include <unistd.h>
#include <signal.h>
#include <chrono>
#include <pthread.h>

#include "modules/ModuleCamera.h"
#include "modules/ModuleGPIO.h"
#include "modules/ModuleRIFT.h"
#include "modules/ModuleDetection.h"
#include "cameraIR/Guide612.h"
#include <jetson-utils/cudaUtility.h>
#include <jetson-utils/glDisplay.h>
#include <jetson-utils/cudaResize.h>
#include <jetson-utils/imageIO.h>

volatile bool signal_recieved = false;
mode keyboard_mode = Fuse;

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

static void* SaveVIS(void* _args)
{
	saveThreadArgs* args = (saveThreadArgs*)_args;
	char filename[256];
	snprintf(filename, sizeof(filename), "/home/cza/cropter/saved_image/VIS/VIS-detected%08d.jpg", args->framenum);
	saveImage(filename, args->frameDetect, args->VIW, args->VIH, IMAGE_RGB8, 90);
	snprintf(filename, sizeof(filename), "/home/cza/cropter/saved_image/VIS/VIS-%08d.jpg", args->framenum);
	saveImage(filename, args->frameVIS, args->VIW, args->VIH, IMAGE_RGB8, 90);
	snprintf(filename, sizeof(filename), "/home/cza/cropter/saved_image/IR/IR-%08d.jpg", args->framenum);
	saveImage(filename, args->frameIR, args->IRW, args->IRH, IMAGE_RGB8, 90);
}

void startSaveThread(saveThreadArgs save_args)
{
	pthread_t save_thread;
	if( pthread_create(&save_thread, NULL, SaveVIS, (void*)&save_args) != 0 )
	{
		printf("[SaveImage Thread] Failed to initialize\n");
	}
	pthread_detach(save_thread);
}

void test()
{
	// Camera
	if( !GuideCamera::Init())
	{
		printf("[Test] failed to initialize IR camera\n");
		return;
	}
	ModuleCamera* cameraIR = new ModuleCamera(NULL, CAM_IR);
    LiteGstCamera* gstVIS = LiteGstCamera::Create(1920, 1080, "/dev/video0");
    if( !gstVIS )
	{
		printf("[Test] failed to initialize VIS camera\n");
		return;
	}
	ModuleCamera* cameraVIS = new ModuleCamera(gstVIS, CAM_VIS);
	// cameraIR->Start();
	cameraVIS->Start();

	// GPIO
    ModuleGPIO* gpio = new ModuleGPIO(cameraIR, cameraVIS);
	// gpio->Start();

	// Registration and RIFT
	ModuleRIFT* rift = new ModuleRIFT(cameraIR, cameraVIS);
	cv::Mat Homography;
	cv::Mat imIR;
	cv::Mat imVIS;
	cv::Mat imIRWarp;
	cv::Mat imIRWarpSplit[3];
	cv::Mat imVISSplit[3];
	cv::Mat imFused;
	void* frameFused = NULL;
	cudaAllocMapped(&frameFused, BYTES_RGB);
	void* frameFusedResized = NULL;
	cudaAllocMapped(&frameFusedResized, cameraVIS->GetWidth()*cameraVIS->GetHeight()*sizeof(uchar3));
	rift->Start();

	// Detection and YOLOv5
	std::string inference_engine = "../../detection/yolov5n.engine"; 
	ModuleDetection* det = new ModuleDetection(inference_engine, rift);
	void* frameVISDetected = NULL;
	cudaAllocMapped(&frameVISDetected, cameraVIS->GetWidth()*cameraVIS->GetHeight()*sizeof(uchar3));
	void* frameVISDetectedResized = NULL;
	cudaAllocMapped(&frameVISDetectedResized, 640*480*sizeof(uchar3));
	void* frameY16 = NULL;

	// Display
    glDisplay* dis = glDisplay::Create(NULL, 1920, 1080);
	void* frameIR = NULL;
	void* frameVIS = NULL;
	void* frameIRResized = NULL;
	cudaAllocMapped(&frameIRResized, 1280*1024*sizeof(uchar3));
	char title[256];

	// Time recording
	int count = 0;
	int runtime = 0;
	int disptime = 0;

	// Image saving
	int framenum = 0;
	void* frameIR_copy = NULL;
	void* frameVIS_copy = NULL;
	void* frameDetect_copy = NULL;
	cudaAllocMapped(&frameIR_copy,  cameraIR->GetWidth()*cameraIR->GetHeight()*sizeof(uchar3));
	cudaAllocMapped(&frameVIS_copy, cameraVIS->GetWidth()*cameraVIS->GetHeight()*sizeof(uchar3));
	cudaAllocMapped(&frameDetect_copy, cameraVIS->GetWidth()*cameraVIS->GetHeight()*sizeof(uchar3));

	// Other settings
	int dummy = 0;
	// gpio->mode_result = Fuse;

	// The main cycle
	while(!signal_recieved)
	{
		auto start = std::chrono::system_clock::now();
		gpio->mode_result = keyboard_mode;

		// Fetch data buffers
		if(!cameraIR->Read(&dummy, (void**)&frameIR, UINT64_MAX))
			printf("[Test] Cannot get IR data\n");
		if(!cameraVIS->Read(&dummy, (void**)&frameVIS, UINT64_MAX))
			printf("[Test] Cannot get VIS data\n");
		if(!GuideCamera::CaptureIRY16(&dummy, &frameY16, UINT64_MAX))
			printf("[Test] Cannot get Y16 data\n");

		// Take different actions depends on mode 
        switch(gpio->mode_result)
		{
			case IR:
			{
				// CUDA(cudaMemcpy(frameVISDetected, frameVIS, cameraVIS->GetWidth()*cameraVIS->GetHeight()*sizeof(uchar3), cudaMemcpyDeviceToDevice));
				// det->Detect(frameVISDetected, frameIR, (short*)frameY16, cameraVIS->GetWidth(), cameraVIS->GetHeight());
				cudaResize((uchar3*)frameIR, cameraIR->GetWidth(), cameraIR->GetHeight(), (uchar3*)frameIRResized, 1280, 1024);
				dis->BeginRender();
				dis->RenderImage(frameIRResized, 1280, 1024, IMAGE_RGB8, 0, 0);
				// dis->RenderImage(frameIR, cameraIR->GetWidth(), cameraIR->GetHeight(), IMAGE_RGB8, 0, 0);
				dis->EndRender();
				break;
			}
			case VI:
			{
				CUDA(cudaMemcpy(frameVISDetected, frameVIS, cameraVIS->GetWidth()*cameraVIS->GetHeight()*sizeof(uchar3), cudaMemcpyDeviceToDevice));
				det->Detect(frameVISDetected, frameIR, (short*)frameY16, cameraVIS->GetWidth(), cameraVIS->GetHeight());
				// cudaResize((uchar3*)frameVISDetected, cameraVIS->GetWidth(), cameraVIS->GetHeight(), (uchar3*)frameVISDetectedResized, 640, 480);
				dis->BeginRender();
				dis->RenderImage(frameVISDetected, cameraVIS->GetWidth(), cameraVIS->GetHeight(), IMAGE_RGB8, 0, 0);
				// dis->RenderImage(frameVISDetectedResized, 640,480, IMAGE_RGB8, 0, 0);
				dis->EndRender();
				break;
			}
			case Fuse:
			{
				// // old version
        		// imIR = cv::Mat(cameraIR->GetHeight(), cameraIR->GetWidth(), CV_8UC3, frameIR);
				// imVIS = cv::Mat(cameraVIS->GetHeight(), cameraVIS->GetWidth(), CV_8UC3, frameVIS);
				// cv::resize(imVIS, imVIS, imIR.size());
				// Homography = rift->getTransMat();
				// cv::warpPerspective(imIR, fused, Homography, imIR.size());
				// cv::addWeighted(imVIS, 0.5, fused, 0.5, 0.0, fused);
				// cudaMemcpy(frameFused, fused.data, BYTES_RGB, cudaMemcpyHostToDevice);
				// cudaResize((uchar3*)frameFused, cameraIR->GetWidth(), cameraIR->GetHeight(), (uchar3*)frameFusedResized, 1280, 1024);

				det->doInference(frameVIS, cameraVIS->GetWidth(), cameraVIS->GetHeight(), det->mRes);
				det->getMaxTemper(frameIR, (short*)frameY16, cameraVIS->GetWidth(), cameraVIS->GetHeight(), false); // 20ms
				imIR = cv::Mat(cameraIR->GetHeight(), cameraIR->GetWidth(), CV_8UC3, frameIR);
				imVIS = cv::Mat(cameraVIS->GetHeight(), cameraVIS->GetWidth(), CV_8UC3, frameVIS).clone();
				cv::resize(imVIS, imVIS, imIR.size(), 0, 0, cv::INTER_NEAREST);
				Homography = rift->getTransMat(); // 2ms
				cv::warpPerspective(imIR, imIRWarp, Homography, imIR.size(), cv::INTER_NEAREST); // 1~2ms
				cv::addWeighted(imVIS, 0.5, imIRWarp, 0.5, 0.0, imFused); // 1ms

				// // use fusion rule
				// cv::cvtColor(imVIS,imVIS,cv::COLOR_RGB2YCrCb);
				// cv::cvtColor(imIRWarp,imIRWarp,cv::COLOR_RGB2YCrCb);
				// split(imVIS,imVISSplit);
				// split(imIRWarp,imIRWarpSplit);
				// imVISSplit[0] = 0.4 * imVISSplit[0] + 0.6 * imIRWarpSplit[0];
				// merge(imVISSplit, 3, imFused);
				// cv::cvtColor(imFused, imFused, cv::COLOR_YCrCb2RGB); // 3ms

				// auto start1 = std::chrono::system_clock::now();
				cudaMemcpy(frameFused, imFused.data, BYTES_RGB, cudaMemcpyHostToHost);
				cudaResize((uchar3*)frameFused, cameraIR->GetWidth(), cameraIR->GetHeight(), (uchar3*)frameFusedResized, cameraVIS->GetWidth(), cameraVIS->GetHeight());
				cudaDeviceSynchronize();
				det->drawBoxLabel(frameFusedResized, cameraVIS->GetWidth(), cameraVIS->GetHeight()); // unstable, 2~50ms
				// auto end1 = std::chrono::system_clock::now();
				// int runtime1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
				// std::cout << "step1 " << runtime1 << std::endl;

				dis->BeginRender();
				// dis->RenderImage(frameFused, GUIDE_CAM_W, GUIDE_CAM_H, IMAGE_RGB8, 0, 0);
				dis->RenderImage(frameFusedResized, cameraVIS->GetWidth(), cameraVIS->GetHeight(), IMAGE_RGB8, 0, 0);
				dis->EndRender(); // unstable, 5~50ms
				break;
			}
			default:
				break;
		};

		// Strat image saving thread
		if(gpio->mode_result == VI && cameraVIS->QuerySaveFlag())
		{
			cudaMemcpy(frameDetect_copy, frameVISDetected, cameraVIS->GetWidth()*cameraVIS->GetHeight()*sizeof(uchar3), cudaMemcpyDeviceToDevice);
			cudaMemcpy(frameVIS_copy, frameVIS, cameraVIS->GetWidth()*cameraVIS->GetHeight()*sizeof(uchar3), cudaMemcpyDeviceToDevice);
			cudaMemcpy(frameIR_copy, frameIR, cameraIR->GetWidth()*cameraIR->GetHeight()*sizeof(uchar3), cudaMemcpyDeviceToDevice);
			saveThreadArgs save_args;
			save_args.frameIR = frameIR_copy;
			save_args.frameVIS = frameVIS_copy;
			save_args.frameDetect = frameDetect_copy;
			save_args.framenum = framenum;
			save_args.VIW = cameraVIS->GetWidth();
			save_args.VIH = cameraVIS->GetHeight();
			save_args.IRW = cameraIR->GetWidth();
			save_args.IRH = cameraIR->GetHeight();

			startSaveThread(save_args);
			framenum++;
			cameraVIS->ClearSaveFlag();
		}

		// Give back buffer pointers
		cameraIR->ReadFinish(frameIR);
		cameraVIS->ReadFinish(frameVIS);
		GuideCamera::CaptureY16Finish(frameY16);

		// Calc time and display
		count++;
		auto end = std::chrono::system_clock::now();
		runtime += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		if(count == 100)
		{
			cameraVIS->SetSaveFlag();
			disptime = runtime / 100;
			runtime = 0;
			count = 0;
		}
		sprintf(title, "Camera Viewer | %.0f FPS",  1000.0 / float(disptime));
        dis->SetTitle(title);


        if( dis->IsClosed() )
            signal_recieved = true;
		
	}

	// Stopping sub threads
	// cameraIR->Stop();
	cameraVIS->Stop();
	// gpio->Stop();
	rift->Stop();
    // cameraIR->Join();
	cameraVIS->Join();
	// gpio->Join();
	rift->Join();

	// Release resources
	// SAFE_DELETE(cameraIR);
	SAFE_DELETE(cameraVIS);
	SAFE_DELETE(gpio);
    SAFE_DELETE(gstVIS);
	// printf("[Test] safe delete3\n");
	SAFE_DELETE(rift)
	SAFE_DELETE(det)
	SAFE_DELETE(dis);
	GuideCamera::DeInit();

	CUDA_FREE_HOST(frameIRResized);
	CUDA_FREE_HOST(frameFused);
	CUDA_FREE_HOST(frameFusedResized);
	CUDA_FREE_HOST(frameVISDetected);
	CUDA_FREE_HOST(frameVISDetectedResized);
	CUDA_FREE_HOST(frameIR_copy);
	CUDA_FREE_HOST(frameVIS_copy);
	CUDA_FREE_HOST(frameDetect_copy);
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
