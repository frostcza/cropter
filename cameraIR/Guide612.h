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

#define BYTES_RGB       (GUIDE_CAM_W * GUIDE_CAM_H * sizeof(uchar3))	//RGB数据一帧的大小
#define BYTES_YUV       (GUIDE_CAM_W*GUIDE_CAM_H*sizeof(unsigned char)*2) //YUV数据一帧的大小,注意原始数据是YUV422 UYVY存储格式，需在回调函数里进行转换
#define BYTES_Y16       (GUIDE_CAM_W * GUIDE_CAM_H * sizeof(uint16_t))	//16位数据一帧的大小
#define NUM_BUF         8												//一次开辟缓存的个数
// #define BYTES_PARAMLINE (GUIDE_CAM_W * sizeof(short)) //参数行大小

/**
 * 用于存储calcTemper函数返回值的结构体
 * 
 * @param maxTemper 最大温度值
 * @param meanTemper 平均温度值
 * @param sectionAverage 最大5%温度的平均值
 * @param maxTemperLoc 最热点所在坐标, 0<=x<639, 0<=y<511
 */
struct TemperResult
{
    float maxTemper; // 最大温度值
    float meanTemper; // 平均温度值
    float sectionAverage; // 最大5%温度的平均值
    cv::Point2i maxTemperLoc; // 最热点所在坐标, 0<=x<639, 0<=y<511
	TemperResult()
	{
		maxTemper = 0.0;
		meanTemper = 0.0;
		sectionAverage = 0.0;
		maxTemperLoc = cv::Point2i(0, 0);
	};
};

enum TemperMode {GLOBAL = 0, HOTSPOT};

// 类内所有成员都是静态的，直接用类名调用，不要实例化这个类
class GuideCamera
{
public:

	// 初始化红外摄像头
    static bool Init();

	// 释放红外摄像头资源
    static bool DeInit();

	/**
	 * @brief 读数据到data, 需配合CaptureFinish使用, 每个新的caller都会被注册入EventsTable。阻塞本线程直到超时或写者调用WakeAllRGBEvent()
	 * @param caller 标识调用者身份的指针
	 * @param data 指向数据缓冲区的二级指针
	 * @param timeout 超时等待时间 单位:秒
	 * @return 成功时返回true
	 */
	static bool CaptureIRRGB(void* caller, void** data, uint64_t timeout);
    static bool CaptureIRY16(void* caller, void** data, uint64_t timeout);

	/**
	 * @brief 读者通过Capture函数取出图像, 用完数据后需调CaptureFinish解开对应缓冲区的读锁
	 * @param p 被Capture函数读出的缓冲区指针, 等于&data
	 * @return 成功时返回true
	 */
	inline static bool CaptureRGBFinish(void* const p) { return mRGBBuff->GiveBack(p, false); }
	inline static bool CaptureY16Finish(void* const p) { return mY16Buff->GiveBack(p, false); }

    // 在frameCallack中将IR图像数据写到缓冲区成功后，调WakeAllEvent()唤醒所有在等待取图的事件
	static bool WakeAllRGBEvent();
    static bool WakeAllY16Event();

	/**
	 * @brief 测温函数 对pts围成的四边形区域 调高德测温接口将Y16数据转温度值
	 * @param y16_data 由CaptureIRY16取得的Y16数据指针, 传入时需强转为(short*)
	 * @param pts 待测区域的四个顶点
	 * @param mode GLOBAL模式会测TemperResult结构中所有参数, HOTSPOT模式只测最大温度及其位置
	 * @return TemperResult结构
	 */
	static TemperResult calcTemper(short* y16_data, cv::Point2i* pts, TemperMode mode);

	/**
	 * @brief 温度校正函数, 进行了测量黑体温度的实验, 对实验数据回归分析得到校正函数
	 * @param measure 相机测得温度
	 * @return 校正后温度
	 */
	static float TemperRevise(float measure);

	/**
	 * @brief 区域生长, 被calcTemper调用, 已有边界的情况下, 在图像input上以pt为种子点进行区域生长, 将待测温位置对应的Y16数据存入output
	 * @param input 待生长图像
	 * @param pt 种子点
	 * @param y16_data 由CaptureIRY16取得的Y16数据指针, 传入时需强转为(short*)
	 * @param output 输出参数, 保存待测位置的Y16数据
	 * @param y16_data_max 输出参数, 本区域最大的Y16值, 用于单独调Y16转温度接口求区域内最大温度
	 * @param max_loc 输出参数, 本区域最大值的坐标, 用于画图
	 */
	static void RegionGrow(cv::Mat& input, cv::Point2i pt, short* y16_data, std::vector<short>& output, short& y16_data_max, cv::Point2i& max_loc);


    static LiteRingBuffer*     mRGBBuff; // 循环队列缓冲区 RGB数据用于显示
    static LiteRingBuffer*     mY16Buff; // 循环队列缓冲区 Y16数据用于测温

	static Mutex			    mMutexRGBEvent; // 用来保护mRGBEventsTable的读写，也即读者的注册
	static std::unordered_map<void*, Event*> mRGBEventsTable; // 存放(caller指针，取图Event)

	static Mutex			    mMutexY16Event;
	static std::unordered_map<void*, Event*> mY16EventsTable;

	static void*                               yuv_on_device; // 进行yuv->rgb转换时yuv数据的Mapped memory
	static bool 					           measureTemper; // 是否在回调函数内开启温度测量
	static int                                 measureTemperCounter; // 回调函数内开启温度测量的counter
	static guide_usb_device_info_t             deviceInfo; // guide coin612初始化所需参数
	static guide_measure_external_param_t      measureExternalParam; // guide coin612温度测量所需参数
	static short*					   		   paramLine; // 相机callback函数会返回一行参数，需保存，测温要用
	static int 								   downsample; // 测温时下采样倍数
	static float							   focalPlane; // 焦平面温度

};

#endif
