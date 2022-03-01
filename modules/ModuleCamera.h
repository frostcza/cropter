#ifndef __MODULE_CAMERA_H__
#define __MODULE_CAMERA_H__

#include "cameraVIS/LiteGstCamera.h"
#include "ModuleTemplate.h"

// 相机类型：红外 or 可见光
enum CAM_TYPE
{
	CAM_IR,
	CAM_VIS,
};

class ModuleCamera : public ModuleTemplate
{
public:

	static ModuleCamera* Create(LiteGstCamera* cam, CAM_TYPE cam_type, uint32_t numBufs=8);

	~ModuleCamera();

	inline uint32_t GetWidth() const { return mWidth; }
	inline uint32_t GetHeight() const { return mHeight; }

	// start thread
	void Start();

	LiteGstCamera*	mGstCam;
	size_t mBytes;

	inline void SetSaveFlag() {save_mutex.Lock(); saveFlag = true; save_mutex.Unlock();}
	inline bool QuerySaveFlag() {save_mutex.Lock(); bool t = saveFlag; save_mutex.Unlock(); return t;}
	inline void ClearSaveFlag() {save_mutex.Lock(); saveFlag = false; save_mutex.Unlock();}

	unsigned char mCamType; // 8bit
private:
	ModuleCamera(LiteGstCamera* cam, CAM_TYPE cam_type, uint32_t width, uint32_t height, uint32_t numBufs);

	bool saveFlag;
	Mutex save_mutex;

	uint32_t mWidth;
	uint32_t mHeight;
};


#endif
