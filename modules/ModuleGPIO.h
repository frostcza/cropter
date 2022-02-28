#ifndef __MODULE_GPIO_H__
#define __MODULE_GPIO_H__

#include <chrono>
#include <thread>
#include <queue>
#include <jetson-utils/Mutex.h>
#include <jetson-utils/Thread.h>
#include "modules/ModuleCamera.h"

enum mode { IR = 0, VI = 1, Fuse = 2, Unknown = 3 };

class ModuleGPIO
{
public:
    ModuleGPIO(ModuleCamera* IRModule, ModuleCamera* VIModule);
    ~ModuleGPIO();

    void Start();

    inline void delay(int s) { std::this_thread::sleep_for(std::chrono::seconds(s)); }

    inline void Join()	{ if(mThreadStarted) {pthread_join(*(modeThread.GetThreadID()), NULL); pthread_join(*(saveFlagThread.GetThreadID()), NULL); mThreadStarted=false;}}
	bool QuerySignal();
	inline void Stop()	{ mMutexRSS.Lock();	mReceivedStopSignal = true;	mMutexRSS.Unlock();	}

    int mode_pin, save_flag_pin;
    struct timeval start_mode, end_mode; // timeval for mode process
    struct timeval start_flag, end_flag; // timeval for save flag process
    int high_voltage_time_mode; // time for mode process
    int high_voltage_time_flag; // time for save flag process
    mode mode_result;
    std::queue<mode> mode_queue;

    ModuleCamera* mIRModule;
    ModuleCamera* mVIModule;

private:
    Thread  modeThread;
    Thread  saveFlagThread;
    
    bool    mThreadStarted;
    bool	mReceivedStopSignal;
	Mutex	mMutexRSS;

};

#endif