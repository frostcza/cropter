#ifndef __MODULE_GPIO_H__
#define __MODULE_GPIO_H__

/* 
 * 使用JestonGPIO库实现PWM信号的解码
 * 遥控器E开关（三段开关）->接收机第9通道->PWM信号->Jetson NX GPIO 11号引脚
 * 遥控器H开关（点击开关）->接收机第10通道->PWM信号->Jetson NX GPIO 12号引脚
 * 可以从高电平持续时间中解码出三段开关状态，可以利用电平跳变检测出是否点击
 * 受图传分辨率限制，同时在屏幕上显示IR，VI和Fused比较难，用三段开关做一个显示模式的切换
 * 点击开关用于发送存图信号
 * https://github.com/pjueon/JetsonGPIO
 */

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