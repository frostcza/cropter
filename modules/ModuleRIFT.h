#ifndef __MODULE_RIFT_H__
#define __MODULE_RIFT_H__

#include "RIFT/rift_no_rotation_invariance.h"
#include "modules/ModuleCamera.h"

class ModuleRIFT
{
public:

    // 外部调用getTransMat()即可获取H矩阵
    // ModuleRIFT的线程Start()之后不断计算H矩阵并setTransMat()尝试更新mtransMat
    // setTransMat()时会调用checkTransMat()检查当前计算出的矩阵是否合理, 判断条件是根据经验设定的
    // checkTransMat() == false时不接受新的H, getTransMat()将向外部返回mtransMatDefault, 即上一个合理的H

	ModuleRIFT(ModuleCamera* IRmodule, ModuleCamera* VISmodule);

	~ModuleRIFT();

	void Start();

    bool checkTransMat(float m[3][3]);

    cv::Mat getTransMat();

    void setTransMat(float transMat[3][3]);

    inline void Join()	{ if(mThreadStarted) {pthread_join(*(mThread.GetThreadID()), NULL); mThreadStarted=false;}	}

	inline bool QuerySignal()  { bool t; mMutexRSS.Lock(); t = mReceivedStopSignal; mMutexRSS.Unlock(); return t;}

	inline void Stop()	{ mMutexRSS.Lock();	mReceivedStopSignal = true;	mMutexRSS.Unlock();	}

    ModuleCamera*		mIRmodule;
	ModuleCamera*		mVISmodule;
    RIFT *mRIFT;

private:

    Thread			mThread;
	bool 			mThreadStarted;
    Mutex           mMutexRSS;
    bool			mReceivedStopSignal;

    cv::Mat mtransMat;
    cv::Mat mtransMatDefault;
    Mutex transMatMutex;
};

#endif