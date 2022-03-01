#ifndef __MODULE_CAMERA_H__
#define __MODULE_CAMERA_H__

#include "cameraVIS/LiteGstCamera.h"
#include <vector>
#include <jetson-utils/Mutex.h>
#include <jetson-utils/Event.h>
#include <jetson-utils/Thread.h>

// 相机类型：红外 or 可见光
enum CAM_TYPE
{
	CAM_IR = 0,
	CAM_VIS,
};

class ModuleCamera
{
public:

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

	bool Read(void* caller, void** data, uint64_t timeout);
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
