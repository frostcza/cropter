/* 使用JestonGPIO库实现PWM信号的解码
 * 遥控器E开关（三段开关）->接收机第9通道->PWM信号->Jetson NX GPIO 11号引脚
 * 遥控器H开关（点击开关）->接收机第10通道->PWM信号->Jetson NX GPIO 12号引脚
 * 可以从高电平持续时间中解码出三段开关状态，可以利用电平跳变检测出是否点击
 * 受图传分辨率限制，同时在屏幕上显示IR，VI和Fused比较难，用三段开关做一个显示模式的切换
 * 点击开关用于实现存图信号的生成
 * https://github.com/pjueon/JetsonGPIO
 */

#include <unistd.h>
#include "ModuleGPIO.h"
#include <signal.h>
#include <sys/time.h>
#include <JetsonGPIO.h>

ModuleGPIO::ModuleGPIO(ModuleCamera* IRModule, ModuleCamera* VIModule)
{
    mIRModule = IRModule;
    mVIModule = VIModule;
    mode_pin = 11; // BOARD pin 11
    save_flag_pin = 12; 
    GPIO::setmode(GPIO::BOARD);
    GPIO::setup(mode_pin, GPIO::IN);
    GPIO::setup(save_flag_pin, GPIO::IN);
}

ModuleGPIO::~ModuleGPIO()
{
    GPIO::cleanup();
    Stop();
}

static void* ModeProcess(void* args)
{
    if(!args)
	{
		printf("[GPIO-ModeProcess] %s:%d | Invaild parameters!\n", __FILE__, __LINE__);
		return NULL;
	}
    ModuleGPIO* moduleGPIO = (ModuleGPIO*)args;

    int countIR = 10;
    int countVI = 0;
    int countFuse = 0;
    for(int i=0; i<10; i++)
    {
        moduleGPIO->mode_queue.push(IR);
    }

    while(!moduleGPIO->QuerySignal())
	{
        // printf("[GPIO-ModeProcess] start to listen...\n");
        GPIO::wait_for_edge(moduleGPIO->mode_pin, GPIO::Edge::RISING);
        gettimeofday(&moduleGPIO->start_mode,NULL);
        GPIO::wait_for_edge(moduleGPIO->mode_pin, GPIO::Edge::FALLING);
        gettimeofday(&moduleGPIO->end_mode,NULL);
        moduleGPIO->high_voltage_time_mode = moduleGPIO->end_mode.tv_usec - moduleGPIO->start_mode.tv_usec;
        // printf("[GPIO-ModeProcess] high voltage time: %d us\n", moduleGPIO->high_voltage_time_mode);

        if(moduleGPIO->high_voltage_time_mode>800 && moduleGPIO->high_voltage_time_mode<=1200)
        {
            moduleGPIO->mode_queue.push(IR);
            countIR++;
        }
        else if(moduleGPIO->high_voltage_time_mode>1200 && moduleGPIO->high_voltage_time_mode<=1600)
        {
            moduleGPIO->mode_queue.push(VI);
            countVI++;
        }
        else if(moduleGPIO->high_voltage_time_mode>1600 && moduleGPIO->high_voltage_time_mode<=2100)
        {
            moduleGPIO->mode_queue.push(Fuse);
            countFuse++;
        }
        else
        {
            continue;
        }

        switch(moduleGPIO->mode_queue.front())
        {
            case IR:
            {
                countIR--;
                break;
            }
            case VI:
            {
                countVI--;
                break;
            }
            case Fuse:
            {
                countFuse--;
                break;
            }
            default:
                break;
        }
        moduleGPIO->mode_queue.pop();

        if(countIR >= 6) moduleGPIO->mode_result = IR;
        if(countVI >= 6) moduleGPIO->mode_result = VI;
        if(countFuse >= 6) moduleGPIO->mode_result = Fuse;
        //printf("[GPIO-ModeProcess] current mode: %d\n", moduleGPIO->mode_result);
        
        // usleep(1000*100);
    }
    printf("[GPIO-ModeProcess] begin to exit...\n");
}

static void* SaveFlagProcess(void* args)
{
    if(!args)
	{
		printf("[GPIO-SaveFlagProcess] %s:%d | Invaild parameters!\n", __FILE__, __LINE__);
		return NULL;
	}
    ModuleGPIO* moduleGPIO = (ModuleGPIO*)args;
    while(!moduleGPIO->QuerySignal())
    {
        GPIO::wait_for_edge(moduleGPIO->save_flag_pin, GPIO::Edge::RISING);
        gettimeofday(&moduleGPIO->start_flag,NULL);
        GPIO::wait_for_edge(moduleGPIO->save_flag_pin, GPIO::Edge::FALLING);
        gettimeofday(&moduleGPIO->end_flag,NULL);
        moduleGPIO->high_voltage_time_flag = moduleGPIO->end_flag.tv_usec - moduleGPIO->start_flag.tv_usec;
        if(moduleGPIO->high_voltage_time_flag>1500 && moduleGPIO->high_voltage_time_flag<=2000)
        {
            moduleGPIO->mIRModule->SetSaveFlag();
            moduleGPIO->mVIModule->SetSaveFlag();
            printf("[GPIO-SaveFlagProcess] received save flag\n");
            moduleGPIO->delay(2);
        }
        else
        {
            moduleGPIO->delay(1);
        }
        
    }
    printf("[GPIO-SaveFlagProcess] begin to exit...\n");
}


bool ModuleGPIO::QuerySignal()
{
	bool t;
	mMutexRSS.Lock();
	t = mReceivedStopSignal;
	mMutexRSS.Unlock();
	return t;
}

void ModuleGPIO::Start()
{
	if(!mThreadStarted)
	{
		mThreadStarted = modeThread.StartThread(ModeProcess, this) && saveFlagThread.StartThread(SaveFlagProcess, this);
	}
}

