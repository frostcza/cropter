/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/*
 * 从jetson-utils/gstCamera裁剪出的可见光摄像头驱动
 * 对建立pipeline的部分进行了简化，直接指定插件序列而不需要协商
 * 用到gsteamer的一整套工具，从v4l2src采入数据，输出到自己做的appsink(mysink)
 * mysink使用gstBufferManager进行数据管理，Capture()函数每次从gstBufferManager取一帧图像
 */

#ifndef __LITE_GST_CAMERA_H__
#define __LITE_GST_CAMERA_H__

#include <gst/gst.h>
#include <string>

#include <jetson-utils/videoSource.h>
#include <jetson-utils/gstBufferManager.h>


struct _GstAppSink;

class LiteGstCamera : public videoSource
{
public:

	// Create函数内会对LiteGstCamera类进行实例化，设置好相关参数并返回实例的指针
	static LiteGstCamera* Create( uint32_t width, uint32_t height, const char* camera=NULL );
	
	// 释放所有资源
	~LiteGstCamera();

	// 打开视频流，在Capture()之前必须先调用Open()
	virtual bool Open();

	// 关闭视频流，析构函数会调用Close()
	virtual void Close();

	// 获取下一帧
	template<typename T> bool Capture( T** image, uint64_t timeout=UINT64_MAX )		{ return Capture((void**)image, imageFormatFromType<T>(), timeout); }

	// 获取下一帧 返回的数据指针，格式(见imageFormat)，超时时间
	virtual bool Capture( void** image, imageFormat format, uint64_t timeout=UINT64_MAX );

	gstBufferManager* mBufferManager;

private:
	static void onEOS(_GstAppSink* sink, void* user_data);
	static GstFlowReturn onPreroll(_GstAppSink* sink, void* user_data);
	static GstFlowReturn onBuffer(_GstAppSink* sink, void* user_data);

	LiteGstCamera( const videoOptions& options );

	bool init(uint32_t width, uint32_t height);

	void checkMsgBus();
	void checkBuffer();

	_GstBus*     mBus;
	_GstAppSink* mAppSink;
	_GstElement* mPipeline;

	std::string  mLaunchStr;

};

#endif
