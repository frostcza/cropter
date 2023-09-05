#ifndef __MODULE_I2C_H__
#define __MODULE_I2C_H__

#include <stdio.h>
#include <jetson-utils/Thread.h>
#include <jetson-utils/Mutex.h>

class ModuleI2C
{
public:
    ModuleI2C();
    ~ModuleI2C();

    void Start();
    inline void Join()	{ if(mThreadStarted) {pthread_join(*(mThread.GetThreadID()), NULL); mThreadStarted=false;}	}
	inline bool QuerySignal()  { bool t; mMutexRSS.Lock(); t = mReceivedStopSignal; mMutexRSS.Unlock(); return t;}
	inline void Stop()	{ mMutexRSS.Lock();	mReceivedStopSignal = true;	mMutexRSS.Unlock();	}

    float mTemperature;
    float mHumidity;

private:
    Thread  mThread;
    bool    mThreadStarted;
    bool	mReceivedStopSignal;
	Mutex	mMutexRSS;

};

#endif