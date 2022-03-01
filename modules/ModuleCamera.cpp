#include <jetson-utils/cudaUtility.h>
#include <jetson-utils/imageFormat.h>
#include <jetson-utils/imageIO.h>
#include <jetson-utils/cudaMappedMemory.h>

#include "cameraIR/Guide612.h"
#include "ModuleCamera.h"

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

// 线程入口函数
// 声明为static可保证其存储在静态内存区，其他线程可以从共享内存区域找到这个函数
// 必须有一个输入参数void* args，代表的是线程创建时，主线程给子线程提供的参数
static void* IRProcess(void* args)
{
	void* imgData = NULL;
	// void* writeData = NULL;

	int framenum = 0;
    int count = 0;
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


static void* VISProcess(void* args)
{
	void* imgData = NULL;
	// void* writeData = NULL;

	int framenum = 0;
    int counter = 0;
    char filename[256];

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
		CUDA(cudaDeviceSynchronize());

		moduleCam->mRGBReadyEvent.Wake();

		if(moduleCam->QuerySaveFlag())
		{
			snprintf(filename, sizeof(filename), "/home/cza/cropter/saved_image/VIS/VIS-%08d.jpg", framenum);
            saveImage(filename, imgData, moduleCam->GetWidth(), moduleCam->GetHeight(), IMAGE_RGB8, 90);
			framenum++;
			moduleCam->ClearSaveFlag();
		}

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
			case CAM_IR:
				mThreadStarted = mthread.StartThread(IRProcess, this);
				break;
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
	// for IR: call CaptureIRRGB
	// for VI: call mGstCam->mBufferManager.mBufferRGB.Peek(RingBuffer::ReadLatest)
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
