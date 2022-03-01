#include <jetson-utils/cudaUtility.h>
#include <jetson-utils/imageFormat.h>
#include <jetson-utils/imageIO.h>
#include <jetson-utils/cudaMappedMemory.h>

#include "cameraIR/Guide612.h"
#include "ModuleCamera.h"

// ModuleCamera的构造函数是private的，不能在外部调用，只有Create是实例化的入口，其内部调用了构造函数
ModuleCamera* ModuleCamera::Create(LiteGstCamera* cam, CAM_TYPE cam_type, uint32_t numBufs)
{
	uint32_t camWidth;
	uint32_t camHeight;
	if(cam_type == CAM_VIS)
	{
		camWidth = cam->GetWidth();
		camHeight = cam->GetHeight();
	}
	else 
	{
		camWidth = GUIDE_CAM_W;
		camHeight = GUIDE_CAM_H;
	}

	ModuleCamera* p = new ModuleCamera(cam, cam_type, camWidth, camHeight, numBufs);
	return p;
}


// 构造函数，先对其父类Moduletemplate初始化，再对自己成员初始化
ModuleCamera::ModuleCamera(LiteGstCamera* cam, CAM_TYPE cam_type, uint32_t width, uint32_t height, uint32_t numBufs):
							ModuleTemplate(sizeof(uchar3)*width*height, numBufs), 
							mGstCam(cam), mWidth(width), mHeight(height), 
							mBytes(sizeof(uchar3)*width*height), mCamType(cam_type), saveFlag(false)
{
}

// 线程入口函数
// 声明为static可保证其存储在静态内存区，其他线程可以从共享内存区域找到这个函数
// 必须有一个输入参数void* args，代表的是线程创建时，主线程
static void* IRProcess(void* args)
{
	void* imgData = NULL;
	void* writeData = NULL;

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
		GuideCamera::CaptureIRRGB(args, &imgData, UINT64_MAX);//获取红外数据。红外数据在高德api的回调函数中实时写入GuideCamera的LiteRingBuffer中。

		if(moduleCam->QuerySaveFlag())
		{
			snprintf(filename, sizeof(filename), "/home/cza/cropter/saved_image/IR/IR-%08d.jpg", framenum);
            saveImage(filename, imgData, GUIDE_CAM_W, GUIDE_CAM_H, IMAGE_RGB8, 90);
			framenum++;
			moduleCam->ClearSaveFlag();
		}
		GuideCamera::CaptureRGBFinish(imgData);

		/* 可以进行一些增强处理，例如将IR VI图像resize到统一大小，目前尚未实现 */

		// 将处理后的图像数据写入IR ModuleCamera的LiteRingBuffer中，此后取图即从此取
		if(!moduleCam->Write(&writeData))
		{
			printf("[Camera-IRProcess] %s:%d | Get data pointer from ringbuffer failed\n", __FILE__, __LINE__);
		}

		// 将处理后的结果写到writeData里
		cudaMemcpy(writeData, imgData, BYTES_RGB, cudaMemcpyDeviceToDevice);

		// Give back buffer
		if(!moduleCam->WriteFinish(writeData))
		{
			printf("[Camera-IRProcess] %s:%d | Give back data to ringbuffer failed\n", __FILE__, __LINE__);
		}
		moduleCam->WakeAllEvent();//每次写完图像数据，立即唤醒所有正在等待取图的Event
	}
	printf("[Camera-IRProcess] begin to exit....\n");
	return NULL;
}


static void* VISProcess(void* args)
{
	void* imgData = NULL;
	void* writeData = NULL;

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
		if( !moduleCam->mGstCam->Capture(&imgData, IMAGE_RGB8, UINT64_MAX))
		{
			printf("[Camera-VISProcess] failed to capture RGB image\n");
		}
		CUDA(cudaDeviceSynchronize());
		if(moduleCam->QuerySaveFlag())
		{
			snprintf(filename, sizeof(filename), "/home/cza/cropter/saved_image/VIS/VIS-%08d.jpg", framenum);
            saveImage(filename, imgData, moduleCam->GetWidth(), moduleCam->GetHeight(), IMAGE_RGB8, 90);
			framenum++;
			moduleCam->ClearSaveFlag();
		}

		/* 可以进行一些增强处理，例如将IR VI图像resize到统一大小，目前尚未实现 */

		if(!moduleCam->Write(&writeData))
		{
			printf("[Camera-VISProcess] %s:%d | Get data pointer from ringbuffer failed\n", __FILE__, __LINE__);
		}

		// 将处理后的结果写到writeData里
		cudaMemcpy(writeData, imgData, sizeof(uchar3)*moduleCam->GetWidth()*moduleCam->GetHeight(), cudaMemcpyDeviceToDevice);

		if(!moduleCam->WriteFinish(writeData))
		{
			printf("[Camera-VISProcess] %s:%d | Give back data to ringbuffer failed\n", __FILE__, __LINE__);
		}
		moduleCam->WakeAllEvent();
	}
	printf("[Camera-VISProcess] begin to exit....\n");
	return NULL;
}

ModuleCamera::~ModuleCamera()
{
	Stop();
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
