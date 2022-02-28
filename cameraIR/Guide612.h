#ifndef __GUIDECAMERA_H__
#define __GUIDECAMERA_H__

#include <bits/stdint-uintn.h>
#include <vector>
#include <unordered_map>
#include <jetson-utils/Mutex.h>
#include <jetson-utils/Event.h>

#include "LiteRingBuffer.h"
// C与C++存在混编问题，对C语言写的文件进行编译时要加上下面这一段
#ifdef __cplusplus
extern "C"
{
#include "include/guideusb2livestream.h"
#include "include/measure.h"
}
#endif

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#define GUIDE_CAM_W     640		//高德相机宽
#define GUIDE_CAM_H     512		//高德相机高

#define BYTES_Y16       (GUIDE_CAM_W * GUIDE_CAM_H * sizeof(uint16_t))	//16位数据一帧的大小
#define BYTES_RGB       (GUIDE_CAM_W * GUIDE_CAM_H * sizeof(uchar3))	//RGB数据一帧的大小
#define BYTES_YUV       (GUIDE_CAM_W*GUIDE_CAM_H*sizeof(unsigned char)*2) //YUV数据一帧的大小,注意原始数据是YUV422 UYVY存储格式，需在回调函数里进行转换

#define NUM_BUF         8												//一次开辟缓存的个数

// 类内所有成员都是静态的，直接用类名调用，不要实例化这个类
class GuideCamera
{
public:
	//初始化
    static bool Init();

	// 释放资源
    static bool DeInit();

    static bool CaptureIRY16(void* caller, void** data, uint64_t timeout);

	// 获取8bit RGB数据
	// 要求调用者传入一个caller指针，用来识别调用者的身份，每新来一个caller都会被注册入mRGBEventsTable，它们都是等待取图的读者
	// 调用CaptureIRRGB时会将该caller对应的Event设置为等待状态，直到WakeAllRGBEvent()被调用，Event被唤醒才读一帧图像到data
    static bool CaptureIRRGB(void* caller, void** data, uint64_t timeout);

	// 针对读者，GetReadBuffer()结束后调用CaptureRGBFinish解开读锁
	inline static bool CaptureY16Finish(void* const p) { return mY16Buff->GiveBack(p, false); }
	inline static bool CaptureRGBFinish(void* const p) { return mRGBBuff->GiveBack(p, false); }

    // 在frameCallack中将IR图像数据写到缓冲区成功后，调用WakeAllRGBEvent()唤醒所有在等待取图的事件
    static bool WakeAllY16Event();
    static bool WakeAllRGBEvent();

	// 测温部分
	static std::vector<float> calcTemper(short* y16_data, cv::Point2i* pts);
	static void RegionGrow(cv::Mat& input, cv::Point2i pt, short* y16_data, std::vector<short>& output);

	// 循环队列缓冲区
    static LiteRingBuffer*     mY16Buff;
    static LiteRingBuffer*     mRGBBuff;

	// static std::vector<Event*>  mY16Events;
	static Mutex			    mMutexY16Event;
	static std::unordered_map<void*, Event*> mY16EventsTable;

	// static std::vector<Event*>  mRGBEvents; // 没有用
	static Mutex			    mMutexRGBEvent; // 用来保护mRGBEventsTable的读写，也即读者的注册
	static std::unordered_map<void*, Event*> mRGBEventsTable; // 存放(caller指针，取图Event)

	static void*                               yuv_on_device; // 进行yuv->rgb转换时yuv数据的显存
	static void* 							   rgb_data_resized; // 进行区域生长时数据的显存
	static bool 					           measureTemper; // 是否进行温度测量
	static int                                 measureTemperCounter; // 不能每帧都做测温，声明一个counter
	static guide_usb_device_info_t             deviceInfo; // guide coin612初始化所需参数
	static guide_measure_external_param_t      measureExternalParam; // guide coin612温度测量所需参数
	static unsigned char*					   paramLine; // 相机callback函数会返回一行参数，需保存，测温模块要用

};

#endif
