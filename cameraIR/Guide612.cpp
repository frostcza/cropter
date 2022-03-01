#include "Guide612.h"
// #include <bits/stdint-uintn.h>

#include <jetson-utils/cudaUtility.h>
#include <jetson-utils/cudaMappedMemory.h>
#include <jetson-utils/cudaColorspace.h>
#include <jetson-utils/cudaResize.h>

#include <algorithm>
#include <numeric>

// static成员变量在.h里声明，在.cpp里初始化
LiteRingBuffer*     				GuideCamera::mY16Buff = NULL;
LiteRingBuffer*     				GuideCamera::mRGBBuff = NULL;
// std::vector<Event*>  				GuideCamera::mY16Events;
Mutex			    				GuideCamera::mMutexY16Event;
std::unordered_map<void*, Event*> 	GuideCamera::mY16EventsTable;
// std::vector<Event*>  				GuideCamera::mRGBEvents;
Mutex			    				GuideCamera::mMutexRGBEvent;
std::unordered_map<void*, Event*> 	GuideCamera::mRGBEventsTable;
void*                               GuideCamera::yuv_on_device = nullptr;
void*                               GuideCamera::rgb_data_resized = nullptr;
bool 					            GuideCamera::measureTemper;
int                                 GuideCamera::measureTemperCounter;
guide_usb_device_info_t             GuideCamera::deviceInfo;
guide_measure_external_param_t      GuideCamera::measureExternalParam;
unsigned char*                      GuideCamera::paramLine = nullptr;

// 没什么用，guide SDK限制，必须写上
int serailCallBack(guide_usb_serial_data_t *pSerialData)
{
    //printf("Serial CallBack\n");
    // int i = 0;
    // for (i = 0;i< pSerialData->serial_recv_data_length;i++)
    // {
    //     if(i== (pSerialData->serial_recv_data_length-1))
    //      {
    //         printf("%x\n",pSerialData->serial_recv_data[i]);
    //     }
    //     else
    //     {
    //         printf("%x ",pSerialData->serial_recv_data[i]);
    //     }
    // }
}

// 打开设备时检查设备状态
int connectStatusCallBack(guide_usb_device_status_e deviceStatus)
{
    if(deviceStatus == DEVICE_CONNECT_OK)
    {
        printf("[cameraIR] VideoStream Capture start...\n");
    }
    else
    {
        printf("[cameraIR] VideoStream Capture end...\n");
    }
}

// 核心函数，是guide camera返回数据的callback函数，摄像头准备好数据以后就会自动回调
// 作为唯一的写者，将数据拷到循环队列以后用GiveBack()归还写锁，调用WakeAllRGBEvent()唤醒所有读者
int frameCallBack(guide_usb_frame_data_t *pVideoData)
{
    void* data;
    data = GuideCamera::mRGBBuff->GetWriteBuffer();
    cudaMemcpy(GuideCamera::yuv_on_device, pVideoData->frame_yuv_data, BYTES_YUV, cudaMemcpyHostToDevice);
    cudaConvertColor(GuideCamera::yuv_on_device, IMAGE_UYVY, data, IMAGE_RGB8, GUIDE_CAM_W, GUIDE_CAM_H);
    GuideCamera::mRGBBuff->GiveBack(data, true);
    GuideCamera::WakeAllRGBEvent();

    void* data_y16;
    data_y16 = GuideCamera::mY16Buff->GetWriteBuffer();
    cudaMemcpy(data_y16, pVideoData->frame_src_data, BYTES_Y16, cudaMemcpyHostToDevice);
    GuideCamera::mY16Buff->GiveBack(data_y16, true);
    GuideCamera::WakeAllY16Event();

    memcpy(GuideCamera::paramLine, pVideoData->paramLine, GUIDE_CAM_W * sizeof(unsigned char));
    
    // 温度测量部分
    // 已改为在外部调用
    // if(GuideCamera::measureTemper)
    // {
    //     GuideCamera::measureTemperCounter++;
    //     float temper;
    //     if(GuideCamera::measureTemperCounter == 50)
    //     {
    //         // 第一个参数是定位到图像最中间的那个像素
    //         guide_measure_convertgray2temper(pVideoData->frame_src_data+pVideoData->frame_width*pVideoData->frame_height/2 + pVideoData->frame_width/2, 
    //             (unsigned char*)pVideoData->paramLine , 1, &GuideCamera::measureExternalParam, &temper);
    //         printf("[cameraIR] center pix Temper:-->:%.1f℃\n",temper);
    //         GuideCamera::measureTemperCounter = 0;
    //     }
    // }

}

// 和guide coin612 SDK有关，初始化设置
bool GuideCamera::Init()
{
    // mY16Events.clear(); 
    // mRGBEvents.clear();

    mY16Buff = new LiteRingBuffer(NUM_BUF, BYTES_Y16);
    mRGBBuff = new LiteRingBuffer(NUM_BUF, BYTES_RGB);

	if( !mY16Buff || !mRGBBuff )
    {
        printf("[LiteRingBuffer] New buffers failed\n");
		return false;
    }
    // 连续打开/关闭摄像头会导致: 在第五次运行时出现libusb_bulk_transfer ret:-7 "operation timeout"，第六次运行时卡死
    // 这应该是高德SDK的一个bug，打开log可获得详细信息
    // guide_usb_setloglevel(15);
    if(guide_usb_initial() < 0)
    {
        printf("[cameraIR] Initialize failed\n");
        return false;
    }

    if(guide_usb_opencommandcontrol((OnSerialDataReceivedCB)serailCallBack) < 0)
    {
        printf("[cameraIR] Open Command Control Failed\n");
        return false;
    }

    cudaAllocMapped(&yuv_on_device, BYTES_YUV);
    cudaAllocMapped(&rgb_data_resized, (GUIDE_CAM_W/4) * (GUIDE_CAM_H/4) * sizeof(uchar3));
    paramLine = (unsigned char*)malloc(GUIDE_CAM_W * sizeof(unsigned char));

    measureTemper = true;
    measureTemperCounter = 0;

    deviceInfo.width = 640;
    deviceInfo.height = 512;
    deviceInfo.video_mode = Y16_PARAM_YUV;

    measureExternalParam.emiss = 98; // 反射率 皮肤典型值98  电力系统元件典型值91
    measureExternalParam.distance = 8; // 距离 单位是米 
    measureExternalParam.relHum = 50; // 湿度
    measureExternalParam.atmosphericTemper = 200; // 环境温度 数值等于摄氏度*10
    measureExternalParam.reflectedTemper = 200; // 反射温度 数值等于摄氏度*10
    measureExternalParam.modifyK = 100; // 参数 K
    measureExternalParam.modifyB = 0; // 参数 B
    
    printf("[cameraIR] coin612 Init succeed\n");
    guide_usb_openstream(&deviceInfo,(OnFrameDataReceivedCB)frameCallBack,(OnDeviceConnectStatusCB)connectStatusCallBack);

    return true;
}

bool GuideCamera::DeInit()
{   
	guide_usb_closestream();
    guide_usb_closecommandcontrol();
    guide_usb_exit();

    delete mY16Buff;
    delete mRGBBuff;

    cudaFreeHost(yuv_on_device);
    cudaFreeHost(rgb_data_resized);

	for(auto it=mY16EventsTable.begin(); it!=mY16EventsTable.end(); ++it)
		delete it->second;

	for(auto it=mRGBEventsTable.begin(); it!=mRGBEventsTable.end(); ++it)
		delete it->second;

    free(paramLine);
    paramLine = nullptr;
    printf("[cameraIR] coin612 Deinit succeed\n");
    return true;
}


bool GuideCamera::WakeAllY16Event()
{
	mMutexY16Event.Lock();
	for(auto it = mY16EventsTable.begin(); it!=mY16EventsTable.end(); ++it)
		it->second->Wake();
	mMutexY16Event.Unlock();
	return true;
}

bool GuideCamera::WakeAllRGBEvent()
{
	mMutexRGBEvent.Lock();
	for(auto it = mRGBEventsTable.begin(); it!=mRGBEventsTable.end(); ++it)
		it->second->Wake();
	mMutexRGBEvent.Unlock();
	return true;
}


bool GuideCamera::CaptureIRY16(void* caller, void** data, uint64_t timeout)
{
	Event* tEvent = NULL;
	if(!caller || !data)
		return false;
	
	// the caller has registered a event, lookup it
	if(mY16EventsTable.count(caller))
	{
		mMutexY16Event.Lock();
		tEvent = mY16EventsTable[caller];
		mMutexY16Event.Unlock();
	}
	else // register a event and add to event table
	{
		tEvent = new Event();
		
		mMutexY16Event.Lock();
		mY16EventsTable.insert(std::pair<void*, Event*>(caller, tEvent));
		mMutexY16Event.Unlock();
	}

	// wait the data is ready with timeout
	// if( !tEvent->Wait(timeout) )
	// 	return false;

	// get the buffer pointer from the ringbuffer
	void* tdata = mY16Buff->GetReadBuffer();
	if(!tdata)
		return false;

	*data = tdata;
	return true;
}


bool GuideCamera::CaptureIRRGB(void* caller, void** data, uint64_t timeout)
{
	Event* tEvent = NULL;
	if(!caller || !data)
		return false;
	
	// the caller has registered a event, lookup it
	if(mRGBEventsTable.count(caller))
	{
		mMutexRGBEvent.Lock();
		tEvent = mRGBEventsTable[caller];
		mMutexRGBEvent.Unlock();
	}
	else // register a event and add to event table
	{
		tEvent = new Event();
		
		mMutexRGBEvent.Lock();
		mRGBEventsTable.insert(std::pair<void*, Event*>(caller, tEvent));
		mMutexRGBEvent.Unlock();
	}

	// wait the data is ready with timeout
	// if( !tEvent->Wait(timeout) )
	// 	return false;

	// get the buffer pointer from the ringbuffer
	void* tdata = mRGBBuff->GetReadBuffer();
	if(!tdata)
		return false;

	*data = tdata;
	return true;
}

// 测温部分
// pts中Point2i.x对应width方向, Point2i.y对应height方向
std::vector<float> GuideCamera::calcTemper(short* y16_data, cv::Point2i* pts)
{
    cv::Mat image = cv::Mat::zeros(GUIDE_CAM_H/4, GUIDE_CAM_W/4, CV_8U);
    std::vector<cv::Point2i> pts_resized(4);
    cv::Point2i pt_begin = cv::Point2i(0,0);
    for(int i = 0; i < 4; i++)
    {
        pts_resized[i].x = pts[i].x / 4;
        pts_resized[i].y = pts[i].y / 4;
        pt_begin.x += pts_resized[i].x;
        pt_begin.y += pts_resized[i].y;
    }
    pt_begin.x /= 4;
    pt_begin.y /= 4;
    cv::polylines(image, pts_resized, true, cv::Scalar(255));
    // cv::fillConvexPoly(image, pts_resized.data(), 4, cv::Scalar(0,255,0));

    std::vector<short> y16_in_region;
    RegionGrow(image, pt_begin, y16_data, y16_in_region);
    int len = y16_in_region.size();
    float temper[len];
    guide_measure_convertgray2temper((short*)y16_in_region.data(), GuideCamera::paramLine, len, &GuideCamera::measureExternalParam, temper);
    std::vector<float> temper_vec(temper, temper + len);

    auto maxPosition = std::max_element(temper_vec.begin(), temper_vec.end());
    float maxTemper = temper_vec[maxPosition - temper_vec.begin()];
    auto minPosition = std::min_element(temper_vec.begin(), temper_vec.end());
    float minTemper = temper_vec[minPosition - temper_vec.begin()];
    double sumTemper = std::accumulate(temper_vec.begin(), temper_vec.end(), 0.0);
	float meanTemper =  sumTemper / (float)temper_vec.size();
    if(minTemper < 0)
    {
        return {0.0, 0.0, 0.0};
    }
    return {maxTemper, minTemper, meanTemper};
}

void GuideCamera::RegionGrow(cv::Mat& input, cv::Point2i pt, short* y16_data, std::vector<short>& output)
{
    cv::Point2i now, temp;
    cv::Mat mark = cv::Mat::zeros(input.rows, input.cols, CV_8U);
    int direction[8][2] = { { -1, -1 }, { 0, -1 }, { 1, -1 }, { 1, 0 }, { 1, 1 }, { 0, 1 }, { -1, 1 }, { -1, 0 } };
    std::vector<cv::Point2i> pointStack;
    pointStack.push_back(pt);
    mark.ptr<uchar>(pt.y)[pt.x] = 1;
    output.push_back(y16_data[pt.y * input.cols * 16 + pt.x * 4]);
    while(!pointStack.empty())
    {
        now = pointStack.back();
        pointStack.pop_back();
        for(int i = 0; i < 8 ; i++)
        {
            temp.x = now.x + direction[i][0];
            temp.y = now.y + direction[i][1];
            if(temp.x < 0 || temp.y < 0 || temp.x > input.cols - 1 || temp.y > input.rows - 1)
				continue;
            if(mark.ptr<uchar>(temp.y)[temp.x] == 0 && input.ptr<uchar>(temp.y)[temp.x] != 255)
            {
                pointStack.push_back(temp);
                mark.ptr<uchar>(temp.y)[temp.x] = 1;
                output.push_back(y16_data[temp.y * input.cols * 16 + temp.x * 4]);
            }
        }
    }
}