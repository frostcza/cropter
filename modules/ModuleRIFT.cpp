#include <opencv2/opencv.hpp>
#include "ModuleRIFT.h"

ModuleRIFT::ModuleRIFT(ModuleCamera* IRmodule, ModuleCamera* VISmodule)
{
    mIRmodule = IRmodule;
    mVISmodule = VISmodule;

    mtransMat = cv::Mat::zeros(3, 3, CV_32F);
    mtransMatDefault = cv::Mat::zeros(3, 3, CV_32F);
    mRIFT = new RIFT(4, 6, 96, 5, 1.0f, cv::Size(mIRmodule->GetWidth(), mIRmodule->GetHeight()));
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
    if ( std::abs(m[2][0]) > 0.001 || std::abs(m[2][1]) > 0.001
				|| m[0][0] <= 0 || m[1][1] <= 0 
				|| std::abs(m[0][2]) > 150 || std::abs(m[1][2]) > 150 )
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
        if(!moduleRIFT->mIRmodule->Read(args, (void**)&IRFrame, UINT64_MAX))
        {
            printf("[RIFT-Process] %s:%d | Cannot get IRFrame!\n", __FILE__, __LINE__);
        }
        CUDA(cudaDeviceSynchronize());
        imIR = cv::Mat(moduleRIFT->mIRmodule->GetHeight(), moduleRIFT->mIRmodule->GetWidth(), CV_8UC3, IRFrame);
        if(!moduleRIFT->mIRmodule->ReadFinish(IRFrame))
        {
            printf("[RIFT-Process] %s:%d | Give back IR pointer failed!\n", __FILE__, __LINE__);
        }

        if(!moduleRIFT->mVISmodule->Read(args, (void**)&VISFrame, UINT64_MAX))
        {
            printf("[RIFT-Process] %s:%d | Cannot get VISFrame!\n", __FILE__, __LINE__);
        }
        CUDA(cudaDeviceSynchronize());
        imVIS = cv::Mat(moduleRIFT->mVISmodule->GetHeight(), moduleRIFT->mVISmodule->GetWidth(), CV_8UC3, VISFrame);
        if(!moduleRIFT->mVISmodule->ReadFinish(VISFrame))
        {
            printf("[RIFT-Process] %s:%d | Give back VIS pointer failed!\n", __FILE__, __LINE__);
        }

        cv::resize(imVIS, imVIS, imIR.size());
        moduleRIFT->mRIFT->Inference(imIR, imVIS, Homography, 0);
        moduleRIFT->setTransMat(Homography);

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