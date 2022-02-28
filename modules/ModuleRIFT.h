#ifndef __MODULE_RIFT_H__
#define __MODULE_RIFT_H__

#include "RIFT/rift_no_rotation_invariance.h"
#include "modules/ModuleCamera.h"

class ModuleRIFT
{
public:

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