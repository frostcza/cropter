#include <jetson-utils/cudaUtility.h>
#include <jetson-utils/imageFormat.h>
#include <jetson-utils/imageIO.h>
#include <jetson-utils/cudaMappedMemory.h>

#include "cameraIR/Guide612.h"
#include "ModuleCamera.h"
#include "modules/ModuleDetection.h"

ModuleCamera::ModuleCamera(LiteGstCamera* cam, CAM_TYPE cam_type)
{
	mReceivedStopSignal = false;
	mThreadStarted = false;

	mCamType = cam_type;
	mSaveFlag = false;
	if(mCamType == CAM_VIS)
	{
		mGstCam = cam;
		mWidth = cam->GetWidth();
		mHeight = cam->GetHeight();
	}
	else if (mCamType == CAM_IR)
	{
		mGstCam = NULL;
		mWidth = GUIDE_CAM_W;
		mHeight = GUIDE_CAM_H;
	}
}

ModuleCamera::~ModuleCamera()
{
	Stop();
}

/* 已取消IRProcess的Start(), 存图功能已移至test.cpp
static void* IRProcess(void* args)
{
	void* imgData = NULL;

	int framenum = 0;
    char filename[256];

	if(!args)
	{
		printf("[Camera-IRProcess] %s:%d | Invaild parameters!\n", __FILE__, __LINE__);
		return NULL;
	}
	ModuleCamera* moduleCam = (ModuleCamera*)args;

	while(!moduleCam->QuerySignal())
	{
		// IR这边是依靠frameCallBack()准备数据的, 所以只在需要存图的时候取出来存一下, 不需要每帧都找LiteRingBuffer要数据
		if(moduleCam->QuerySaveFlag())
		{
			if(!GuideCamera::CaptureIRRGB(args, &imgData, UINT64_MAX))
			{
				printf("[Camera-IRProcess] failed to capture IR image\n");
			}
			snprintf(filename, sizeof(filename), "/home/cza/cropter/saved_image/IR/IR-%08d.jpg", framenum);
            saveImage(filename, imgData, GUIDE_CAM_W, GUIDE_CAM_H, IMAGE_RGB8, 90);
			framenum++;
			moduleCam->ClearSaveFlag();
			GuideCamera::CaptureRGBFinish(imgData);
		}

	}
	printf("[Camera-IRProcess] begin to exit....\n");
	return NULL;
}
*/

static void* VISProcess(void* args)
{
	void* imgData = NULL;
	if(!args)
	{
		printf("[Camera-VISProcess] %s:%d | Invaild parameters!\n", __FILE__, __LINE__);
		return NULL;
	}
	ModuleCamera* moduleCam = (ModuleCamera*)args;

	while(!moduleCam->QuerySignal())
	{
		// mGstCam->Capture()会调mBufferManager->Dequeue()
		// mBufferManager->Dequeue()会从mBufferYUV取一帧, 转成RGB后存入mBufferRGB, 把存入的地址返回
		// i.e., 每次调mGstCam->Capture()时会生产一帧RGB数据并返回, 故此处开一个线程不断调用Capture是有必要的, 它实际上在生产数据
		// 另, mBufferYUV存在的原因是: gst的自定义appsink产出的数据格式为YUV,
		// gst pipeline上的YUV数据在mBufferManager->Enqueue()时被拷贝到mBufferYUV, 但没做convertColor
		if( !moduleCam->mGstCam->Capture(&imgData, IMAGE_RGB8, UINT64_MAX))
		{
			printf("[Camera-VISProcess] failed to capture RGB image\n");
		}
		// moduleCam->mRGBReadyEvent.Wake();

		// 存图功能已移至test.cpp
	}
	printf("[Camera-VISProcess] begin to exit....\n");
	return NULL;
}

void ModuleCamera::Start()
{
	if(!mThreadStarted)
	{
		switch (mCamType) 
		{
			// case CAM_IR:
			// 	mThreadStarted = mthread.StartThread(IRProcess, this);
			// 	break;
			case CAM_VIS:
				mThreadStarted = mthread.StartThread(VISProcess, this);
				break;
			default:
				break;
		}
	}
}

bool ModuleCamera::QuerySignal()
{
	bool t;
	mMutexRSS.Lock();
	t = mReceivedStopSignal;
	mMutexRSS.Unlock();
	return t;
}


bool ModuleCamera::Read(void* caller, void** data, uint64_t timeout)
{
	if(mCamType == CAM_IR)
	{
		return GuideCamera::CaptureIRRGB(caller, data, timeout);
	}
	else if(mCamType == CAM_VIS)
	{
		// if( !mRGBReadyEvent.Wait(timeout))
		// 	return false;
		
		*data = mGstCam->mBufferManager->mBufferRGB.Peek(RingBuffer::ReadLatest);
		return true;
	}
}

bool ModuleCamera::ReadFinish(void* const p)
{
	if(mCamType == CAM_IR)
	{
		return GuideCamera::CaptureRGBFinish(p);
	}
}
