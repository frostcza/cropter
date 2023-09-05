#include "sht/shtc3.h"
#include "ModuleI2C.h"
#include <unistd.h>
#include "cameraIR/Guide612.h"

ModuleI2C::ModuleI2C()
{
    sensirion_i2c_init();
    int retry = 5;
    while (shtc1_probe() != STATUS_OK && retry) 
    {
        printf("[ModuleI2C] SHT sensor init failed\n");
        sensirion_sleep_usec(1000000); /* sleep 1s */
        retry--;
    }
    printf("[ModuleI2C] SHT sensor init success\n");

}

ModuleI2C::~ModuleI2C()
{
    shtc1_sleep();
    sensirion_i2c_release();
}


static void* I2CProcess(void* args)
{
    if(!args)
	{
		printf("[I2C-Process] %s:%d | Invaild parameters!\n", __FILE__, __LINE__);
		return NULL;
	}
    ModuleI2C* moduleI2C = (ModuleI2C*)args;

    while(!moduleI2C->QuerySignal())
	{
        int32_t temperature, humidity;
        int8_t ret = shtc1_measure_blocking_read(&temperature, &humidity);
        if (ret == STATUS_OK) 
        {
            moduleI2C->mTemperature = temperature / 1000.0f;
            moduleI2C->mHumidity = humidity / 1000.0f;
            // printf("[I2C-Process] Temperature: %0.2f℃\t", temperature / 1000.0f);
            // printf("Humidity: %0.2f%%\n", humidity / 1000.0f);
            GuideCamera::measureExternalParam.atmosphericTemper = (int) temperature / 100;
            GuideCamera::measureExternalParam.relHum = (int) humidity / 1000;
        }
        else 
        {
            printf("[I2C-Process] Measurement failed\n");
        }
        sleep(10);
    }
    printf("[I2C-Process] begin to exit...\n");
}

void ModuleI2C::Start()
{
	if(!mThreadStarted)
	{
		mThreadStarted = mThread.StartThread(I2CProcess, this);
	}
}
