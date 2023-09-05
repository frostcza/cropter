#ifndef __MODULE_CAMERA_H__
#define __MODULE_CAMERA_H__

#include "cameraVIS/LiteGstCamera.h"
#include <vector>
#include <jetson-utils/Mutex.h>
#include <jetson-utils/Event.h>
#include <jetson-utils/Thread.h>

enum CAM_TYPE
{
	CAM_IR = 0,
	CAM_VIS,
};

class ModuleCamera
{
public:

	/**
	 * @brief ModuleCamera Constructor
	 * @param cam Pointer of LiteGstCamera if cam_type is CAM_VIS and NULL if cam_type is CAM_IR
	 * @param cam_type CAM_IR or CAM_VIS
	 */
	ModuleCamera(LiteGstCamera* cam, CAM_TYPE cam_type);
	~ModuleCamera();

	inline uint32_t GetWidth() const { return mWidth; }
	inline uint32_t GetHeight() const { return mHeight; }

	inline void SetSaveFlag() {mSaveMutex.Lock(); mSaveFlag = true; mSaveMutex.Unlock();}
	inline bool QuerySaveFlag() {mSaveMutex.Lock(); bool t = mSaveFlag; mSaveMutex.Unlock(); return t;}
	inline void ClearSaveFlag() {mSaveMutex.Lock(); mSaveFlag = false; mSaveMutex.Unlock();}

	void Start();
	inline void Join()	{ if(mThreadStarted) {pthread_join(*(mthread.GetThreadID()), NULL); mThreadStarted=false;}	}
	bool QuerySignal();
	inline void Stop()	{ mMutexRSS.Lock();	mReceivedStopSignal = true;	mMutexRSS.Unlock();	}

	/**
	 * @brief 对两种摄像头读图接口的封装
	 * @param caller 标识调用者身份的指针
	 * @param data 指向数据缓冲区的二级指针
	 * @param timeout 超时等待时间 单位:秒
	 * @return 成功时返回true
	 */
	bool Read(void* caller, void** data, uint64_t timeout);

	/**
	 * @brief 用完数据后需调ReadFinish解开对应缓冲区的读锁
	 * @param p 被Read函数读出的缓冲区指针, 等于&data
	 * @return 成功时返回true
	 */
	bool ReadFinish(void* const p);

	LiteGstCamera* mGstCam;
	Event mRGBReadyEvent;

private:

	unsigned char mCamType;

	uint32_t mWidth;
	uint32_t mHeight;

	bool mSaveFlag;
	Mutex mSaveMutex;

	bool mThreadStarted;
	bool mReceivedStopSignal;
	Mutex mMutexRSS;

	Thread mthread;

};


#endif
