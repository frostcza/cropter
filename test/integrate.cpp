#include "integrate.h"

#include <jetson-utils/cudaUtility.h>
#include <jetson-utils/cudaResize.h>
#include <jetson-utils/imageIO.h>
#include <chrono>
#include <pthread.h>


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

void Integrate::startSaveThread(saveThreadArgs save_args)
{
	pthread_t save_thread;
	if( pthread_create(&save_thread, NULL, SaveVIS, (void*)&save_args) != 0 )
	{
		printf("[SaveImage Thread] Failed to initialize\n");
	}
	pthread_detach(save_thread);
}

Integrate::Integrate(RunOption opt)
{
    mOption = opt;
    MemoryAlloc();
    Init();
    count = 0;
	runtime = 0;
	disptime = 0;
    framenum = 0;
    dummy = 0;
}

Integrate::~Integrate()
{
    // Stopping sub threads
	cameraVIS->Stop();
	rift->Stop();
    i2c->Stop();
	cameraVIS->Join();
	rift->Join();
    i2c->Join();

    if(mOption.use_GPIO)
    {
        gpio->Stop();
	    gpio->Join();
    }

	// Release resources
	SAFE_DELETE(cameraVIS);
	SAFE_DELETE(gpio);
    SAFE_DELETE(i2c);
    SAFE_DELETE(gstVIS);
	SAFE_DELETE(rift)
	SAFE_DELETE(det)
	SAFE_DELETE(dis);
	GuideCamera::DeInit();

    CUDA_FREE_HOST(frameVISDetected);
    CUDA_FREE_HOST(frameFused);
    if(mOption.shrink_picture)
    {
        CUDA_FREE_HOST(frameVISDetectedSmall);
    }
    else
    {
        CUDA_FREE_HOST(frameIRLarge);
        CUDA_FREE_HOST(frameFusedLarge);
    }
	CUDA_FREE_HOST(frameIR_copy);
	CUDA_FREE_HOST(frameVIS_copy);
	CUDA_FREE_HOST(frameDetect_copy);

    frameIR = NULL;
	frameVIS = NULL;
    frameY16 = NULL;

    frameFused = NULL;
    frameFusedLarge = NULL;

    frameVISDetected = NULL;
    frameVISDetectedSmall = NULL;
    
	frameIRLarge = NULL;

	frameIR_copy = NULL;
	frameVIS_copy = NULL;
	frameDetect_copy = NULL;
}

void Integrate::MemoryAlloc()
{
    frameIR = NULL;
	frameVIS = NULL;
    frameY16 = NULL;

    frameFused = NULL;
    frameFusedLarge = NULL;

    frameVISDetected = NULL;
    frameVISDetectedSmall = NULL;
    
	frameIRLarge = NULL;

	frameIR_copy = NULL;
	frameVIS_copy = NULL;
	frameDetect_copy = NULL;

    cudaAllocMapped(&frameVISDetected, cameraVIS_W*cameraVIS_H*sizeof(uchar3));
    cudaAllocMapped(&frameFused, BYTES_RGB);
	

    if(mOption.shrink_picture)
    {
        cudaAllocMapped(&frameVISDetectedSmall, 640*480*sizeof(uchar3));
    }
    else
    {
        cudaAllocMapped(&frameIRLarge, 1280*1024*sizeof(uchar3));
        cudaAllocMapped(&frameFusedLarge, cameraVIS_W*cameraVIS_H*sizeof(uchar3));
    }

	cudaAllocMapped(&frameIR_copy, cameraIR_W*cameraIR_H*sizeof(uchar3));
	cudaAllocMapped(&frameVIS_copy, cameraVIS_W*cameraVIS_H*sizeof(uchar3));
	cudaAllocMapped(&frameDetect_copy, cameraVIS_W*cameraVIS_H*sizeof(uchar3));

}

bool Integrate::Init()
{
    // Camera
    if(!GuideCamera::Init())
	{
		printf("[Integrate] failed to initialize IR camera\n");
		std::abort();
	}
    cameraIR = new ModuleCamera(NULL, CAM_IR);
    gstVIS = LiteGstCamera::Create(cameraVIS_W, cameraVIS_H, "/dev/video0");
    if(!gstVIS )
	{
		printf("[Integrate] failed to initialize VIS camera\n");
		std::abort();
	}
    cameraVIS = new ModuleCamera(gstVIS, CAM_VIS);
	cameraVIS->Start();

	// GPIO and Remote Control
    gpio = new ModuleGPIO(cameraIR, cameraVIS);
    if(mOption.use_GPIO)
    {
	    gpio->Start();
    }

    // I2C and Temperature/Humidity Mesurement
    i2c = new ModuleI2C();
    i2c->Start();

    // Registration and RIFT
	rift = new ModuleRIFT(cameraIR, cameraVIS);
	rift->Start();

    // Detection and YOLOv5
	det = new ModuleDetection(inference_engine, rift);

    // Display
    dis = glDisplay::Create(NULL, cameraVIS_W, cameraVIS_H);

    return true;
}

void Integrate::mainLoop()
{
    auto start = std::chrono::system_clock::now();
    
    // Fetch data buffers
    if(!cameraIR->Read(&dummy, (void**)&frameIR, UINT64_MAX))
        printf("[Integrate] Cannot get IR data\n");
    if(!cameraVIS->Read(&dummy, (void**)&frameVIS, UINT64_MAX))
        printf("[Integrate] Cannot get VIS data\n");
    if(!GuideCamera::CaptureIRY16(&dummy, &frameY16, UINT64_MAX))
        printf("[Integrate] Cannot get Y16 data\n");

    // Take different actions depends on mode 
    switch(gpio->mode_result)
    {
        case IR:
        {
            dis->BeginRender();
            if(mOption.shrink_picture)
            {
                // CUDA(cudaMemcpy(frameVISDetected, frameVIS, cameraVIS_W*cameraVIS_H*sizeof(uchar3), cudaMemcpyDeviceToDevice));
                // det->Detect(frameVISDetected, frameIR, (short*)frameY16, cameraVIS_W, cameraVIS_H);
                dis->RenderImage(frameIR, cameraIR_W, cameraIR_H, IMAGE_RGB8, 0, 0);
            }
            else
            {
                cudaResize((uchar3*)frameIR, cameraIR_W, cameraIR_H, (uchar3*)frameIRLarge, 1280, 1024);
                dis->RenderImage(frameIRLarge, 1280, 1024, IMAGE_RGB8, 320, 0);
            }
            dis->EndRender();
            break;
        }
        case VI:
        {
            CUDA(cudaMemcpy(frameVISDetected, frameVIS, cameraVIS_W*cameraVIS_H*sizeof(uchar3), cudaMemcpyDeviceToDevice));
            det->Detect(frameVISDetected, frameIR, (short*)frameY16, cameraVIS_W, cameraVIS_H);

            dis->BeginRender();
            if(mOption.shrink_picture)
            {
                cudaResize((uchar3*)frameVISDetected, cameraVIS_W, cameraVIS_H, (uchar3*)frameVISDetectedSmall, 640, 480);
                dis->RenderImage(frameVISDetectedSmall, 640,480, IMAGE_RGB8, 0, 0);
            }
            else
            {
                dis->RenderImage(frameVISDetected, cameraVIS_W, cameraVIS_H, IMAGE_RGB8, 0, 0);
            }
            dis->EndRender();
            break;
        }
        case Fuse:
        {
            det->doInference(frameVIS, cameraVIS_W, cameraVIS_H, det->mRes);
            det->getMaxTemper(frameIR, (short*)frameY16, cameraVIS_W, cameraVIS_H, false); // 20ms
            imIR = cv::Mat(cameraIR_H, cameraIR_W, CV_8UC3, frameIR);
            imVIS = cv::Mat(cameraVIS_H, cameraVIS_W, CV_8UC3, frameVIS).clone();
            cv::resize(imVIS, imVIS, imIR.size(), 0, 0, cv::INTER_NEAREST);
            Homography = rift->getTransMat(); // 2ms
            cv::warpPerspective(imIR, imIRWarp, Homography, imIR.size(), cv::INTER_NEAREST); // 1~2ms
            // imFused = cv::Mat(cameraIR_H, cameraIR_W, CV_8UC3, frameFused);
            cv::addWeighted(imVIS, 0.5, imIRWarp, 0.5, 0.0, imFused); // 1ms
            cudaMemcpy(frameFused, imFused.data, BYTES_RGB, cudaMemcpyHostToHost);

            // // use fusion rule
            // cv::cvtColor(imVIS,imVIS,cv::COLOR_RGB2YCrCb);
            // cv::cvtColor(imIRWarp,imIRWarp,cv::COLOR_RGB2YCrCb);
            // split(imVIS,imVISSplit);
            // split(imIRWarp,imIRWarpSplit);
            // imVISSplit[0] = 0.4 * imVISSplit[0] + 0.6 * imIRWarpSplit[0];
            // merge(imVISSplit, 3, imFused);
            // cv::cvtColor(imFused, imFused, cv::COLOR_YCrCb2RGB); // 3ms

            dis->BeginRender();
            if(mOption.shrink_picture)
            {
                dis->RenderImage(frameFused, GUIDE_CAM_W, GUIDE_CAM_H, IMAGE_RGB8, 0, 0);
            }
            else
            {
                cudaResize((uchar3*)frameFused, cameraIR_W, cameraIR_H, (uchar3*)frameFusedLarge, cameraVIS_W, cameraVIS_H);
                cudaDeviceSynchronize();
                det->drawBoxLabel(frameFusedLarge, cameraVIS_W, cameraVIS_H); // unstable, 2~50ms
                dis->RenderImage(frameFusedLarge, cameraVIS_W, cameraVIS_H, IMAGE_RGB8, 0, 0);
            }
            dis->EndRender(); // unstable, 5~50ms
            break;
        }
        default:
            break;
    };

    // Strat image saving thread
    if(gpio->mode_result == VI && cameraVIS->QuerySaveFlag())
    {
        cudaMemcpyAsync(frameDetect_copy, frameVISDetected, cameraVIS_W*cameraVIS_H*sizeof(uchar3), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(frameVIS_copy, frameVIS, cameraVIS_W*cameraVIS_H*sizeof(uchar3), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(frameIR_copy, frameIR, cameraIR_W*cameraIR_H*sizeof(uchar3), cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        saveThreadArgs save_args;
        save_args.frameIR = frameIR_copy;
        save_args.frameVIS = frameVIS_copy;
        save_args.frameDetect = frameDetect_copy;
        save_args.framenum = framenum;
        save_args.VIW = cameraVIS_W;
        save_args.VIH = cameraVIS_H;
        save_args.IRW = cameraIR_W;
        save_args.IRH = cameraIR_H;

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
        unsigned char cmd[] = {0x55, 0xAA, 0x07, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00, 0x87, 0xF0};
        guide_usb_sendcommand(cmd, 12);
        if(!mOption.use_GPIO)
            cameraVIS->SetSaveFlag();
        disptime = runtime / 100;
        runtime = 0;
        count = 0;
    }
    sprintf(title, "Camera Viewer | %.0f FPS",  1000.0 / float(disptime));
    dis->SetTitle(title);

}
