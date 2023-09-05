#include <opencv2/opencv.hpp>
#include "ModuleRIFT.h"
#include <unistd.h>

ModuleRIFT::ModuleRIFT(ModuleCamera* IRmodule, ModuleCamera* VISmodule)
{
    mIRmodule = IRmodule;
    mVISmodule = VISmodule;

    mtransMat = cv::Mat::zeros(3, 3, CV_32F);
    mtransMatDefault = cv::Mat::zeros(3, 3, CV_32F);
    mRIFT = new RIFT(4, 6, 96, 5, cv::Size(mIRmodule->GetWidth(), mIRmodule->GetHeight()));
    float matDefault[3][3] = {1.15, 0, 48, 0, 1.25, -45, 0, 0, 1};
    for(int i=0;i<3;i++)
    {
        for(int j=0;j<3;j++)
        {
            mtransMat.at<float>(i,j) = matDefault[i][j];
            mtransMatDefault.at<float>(i,j) = matDefault[i][j];
        }
    }

}

ModuleRIFT::~ModuleRIFT()
{
    Stop();
}

bool ModuleRIFT::checkTransMat(float m[3][3])
{
    	// Homography matrix evaluation: reject wrong result according to experience
		// https://zhuanlan.zhihu.com/p/74597564
		// 	transMat =	[a	 b	 c;
		//     			 d	 e	 f;
		// 				 g	 h	 i]
		// 1. 认为两个相机满足仿射变换而不是单应变换，g,h应接近于0（使用findHomography的原因是opencv没提供仿射接口）
		// 2. 旋转不可能大于90度，且不存在翻转，a,e应同时大于零（根据经验应大于0.5）
		// 3. 图像本身是640x512大小，c,f是平移，不可能很大，一般绝对值在150以内
		// 4. 由于我们的VI摄像头在左，IR摄像头在右，所以把IR往VI上配时，一般X方向上的位移c>0
        // 5. e:a = W方向放大的比例:H方向放大的比例, 即(1920/640) : (1080/512) = 1.422, 放宽到1.2和1.6之间
        // 6. b,d的绝对值小于0.10, 限制旋转角度较小
    if ( std::abs(m[2][0]) > 0.001 || std::abs(m[2][1]) > 0.001
				|| m[0][0] <= 0.5 || m[1][1] <= 0.5 
				|| std::abs(m[0][2]) > 150 || std::abs(m[1][2]) > 150 
                || m[1][1] / m[0][0] < 1.2 || m[1][1] / m[0][0] > 1.6
                || std::abs(m[0][1]) > 0.10 || std::abs(m[1][0]) > 0.10)
    {
        return false;
    }
    else
    {
        return true;
    }
}

cv::Mat ModuleRIFT::getTransMat()
{
    cv::Mat transMatReturn = cv::Mat::zeros(3, 3, CV_32F);
    transMatMutex.Lock();
    transMatReturn = mtransMat.clone();
    transMatMutex.Unlock();
    return transMatReturn;
}

void ModuleRIFT::setTransMat(float transMat[3][3])
{
    transMatMutex.Lock();
    if(checkTransMat(transMat))
    {
        // printf("accept!\n");
        mtransMat = cv::Mat(3, 3, CV_32F, transMat);
        mtransMatDefault = mtransMat.clone();
    }
    else
    {
        // printf("reject!\n");
        mtransMat = mtransMatDefault.clone();
    }
    transMatMutex.Unlock();
}

static void* RIFTProcess(void* args)
{
    if(!args)
	{
		printf("[RIFT-Process] %s:%d | Invaild parameters!\n", __FILE__, __LINE__);
		return NULL;
	}
    ModuleRIFT* moduleRIFT = (ModuleRIFT*)args;

    uchar3* IRFrame = NULL;
	uchar3* VISFrame = NULL;
    cv::Mat imIR;
    cv::Mat imVIS;
    float Homography[3][3];

    while(!moduleRIFT->QuerySignal())
    {
        // auto start = std::chrono::system_clock::now();

        if(!moduleRIFT->mIRmodule->Read(args, (void**)&IRFrame, UINT64_MAX))
        {
            printf("[RIFT-Process] %s:%d | Cannot get IRFrame!\n", __FILE__, __LINE__);
        }
        CUDA(cudaDeviceSynchronize());
        imIR = cv::Mat(moduleRIFT->mIRmodule->GetHeight(), moduleRIFT->mIRmodule->GetWidth(), CV_8UC3, IRFrame).clone();
        if(!moduleRIFT->mIRmodule->ReadFinish(IRFrame))
        {
            printf("[RIFT-Process] %s:%d | Give back IR pointer failed!\n", __FILE__, __LINE__);
        }

        if(!moduleRIFT->mVISmodule->Read(args, (void**)&VISFrame, UINT64_MAX))
        {
            printf("[RIFT-Process] %s:%d | Cannot get VISFrame!\n", __FILE__, __LINE__);
        }
        CUDA(cudaDeviceSynchronize());
        imVIS = cv::Mat(moduleRIFT->mVISmodule->GetHeight(), moduleRIFT->mVISmodule->GetWidth(), CV_8UC3, VISFrame).clone();
        if(!moduleRIFT->mVISmodule->ReadFinish(VISFrame))
        {
            printf("[RIFT-Process] %s:%d | Give back VIS pointer failed!\n", __FILE__, __LINE__);
        }

        cv::resize(imVIS, imVIS, imIR.size());
        moduleRIFT->mRIFT->Inference(imIR, imVIS, Homography, 0);
        moduleRIFT->setTransMat(Homography);
        sleep(2);
        // auto end = std::chrono::system_clock::now();
        // int runtime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        // std::cout << "RIFT 1 frame: " << runtime << std::endl;
    }
    printf("[RIFT-Process] begin to exit...\n");
}


void ModuleRIFT::Start()
{
	if(!mThreadStarted)
	{
		mThreadStarted = mThread.StartThread(RIFTProcess, this);
	}
}
