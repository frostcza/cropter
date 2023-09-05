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

#include "LiteGstCamera.h"
#include "NvInfer.h"
#include <sstream> 
#include <unistd.h>
#include <string.h>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>

#include <jetson-utils/gstUtility.h>
#include <jetson-utils/cudaColorspace.h>
#include <jetson-utils/filesystem.h>
#include <jetson-utils/logging.h>

// constructor
LiteGstCamera::LiteGstCamera( const videoOptions& options ) : videoSource(options)
{	
	mAppSink   = NULL;
	mBus       = NULL;
	mPipeline  = NULL;	
	mBufferManager = new gstBufferManager(&mOptions);
	mBufferManager->mBufferRGB.SetThreaded(true);
}

// destructor	
LiteGstCamera::~LiteGstCamera()
{
	Close();
	if( mAppSink != NULL )
	{
		gst_object_unref(mAppSink);
		mAppSink = NULL;
	}
	if( mBus != NULL )
	{
		gst_object_unref(mBus);
		mBus = NULL;
	}
	if( mPipeline != NULL )
	{
		gst_object_unref(mPipeline);
		mPipeline = NULL;
	}
	SAFE_DELETE(mBufferManager);
}


// Create
LiteGstCamera* LiteGstCamera::Create( uint32_t width, uint32_t height, const char* camera )
{
	videoOptions opt;
	opt.resource = camera;
	opt.width    = width;
	opt.height   = height;
	opt.numBuffers = 8;
	opt.zeroCopy = true;
	opt.deviceType = videoOptions::DEVICE_V4L2;
	opt.ioType   = videoOptions::INPUT;
	opt.codec = videoOptions::CODEC_MJPEG;

	if( !gstreamerInit() )
	{
		LogError(LOG_GSTREAMER "failed to initialize gstreamer API\n");
		return NULL;
	}

    LiteGstCamera* cam = new LiteGstCamera(opt);

    if( !cam )
		return NULL;

	if( !cam->init(width, height) )
	{
		LogError(LOG_GSTREAMER "LiteGstCamera -- failed to create device %s\n", cam->GetResource().c_str());
		return NULL;
	}
	
	LogInfo(LOG_GSTREAMER "LiteGstCamera successfully created device %s\n", cam->GetResource().c_str()); 
	return cam;
}

// init
bool LiteGstCamera::init(uint32_t width, uint32_t height)
{
	GError* err = NULL;

	// init launch str
	std::ostringstream ss;
    ss << "v4l2src device=/dev/video0 ! image/jpeg, width="<<width<<", hight="<<height<<", framerate=30/1 ! jpegdec ! video/x-raw ! appsink name=mysink";  
	mLaunchStr = ss.str();
	LogInfo(LOG_GSTREAMER "LiteGstCamera pipeline string:\n");
	LogInfo(LOG_GSTREAMER "%s\n", mLaunchStr.c_str());

	// launch pipeline
	mPipeline = gst_parse_launch(mLaunchStr.c_str(), &err);
	if( err != NULL )
	{
		LogError(LOG_GSTREAMER "LiteGstCamera failed to create pipeline\n");
		LogError(LOG_GSTREAMER "   (%s)\n", err->message);
		g_error_free(err);
		return false;
	}

	GstPipeline* pipeline = GST_PIPELINE(mPipeline);
	if( !pipeline )
	{
		LogError(LOG_GSTREAMER "LiteGstCamera failed to cast GstElement into GstPipeline\n");
		return false;
	}	

	// retrieve pipeline bus
	mBus = gst_pipeline_get_bus(pipeline);
	if( !mBus )
	{
		LogError(LOG_GSTREAMER "LiteGstCamera failed to retrieve GstBus from pipeline\n");
		return false;
	}

	// get the appsrc
	GstElement* appsinkElement = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
	GstAppSink* appsink = GST_APP_SINK(appsinkElement);
	if( !appsinkElement || !appsink)
	{
		LogError(LOG_GSTREAMER "LiteGstCamera failed to retrieve AppSink element from pipeline\n");
		return false;
	}
	mAppSink = appsink;
	
	// setup callbacks
	GstAppSinkCallbacks cb;
	memset(&cb, 0, sizeof(GstAppSinkCallbacks));
	cb.eos         = onEOS;
	cb.new_preroll = onPreroll;
	cb.new_sample  = onBuffer;
	gst_app_sink_set_callbacks(mAppSink, &cb, (void*)this, NULL);
	
	// disable looping for cameras
	mOptions.loop = 0;	

	// set device flags
	if( mOptions.resource.protocol == "csi" )
		mOptions.deviceType = videoOptions::DEVICE_CSI;
	else if( mOptions.resource.protocol == "v4l2" )
		mOptions.deviceType = videoOptions::DEVICE_V4L2;

	return true;
}


// onEOS
void LiteGstCamera::onEOS(_GstAppSink* sink, void* user_data)
{
	LogWarning(LOG_GSTREAMER "LiteGstCamera -- end of stream (EOS)\n");
}

// onPreroll
GstFlowReturn LiteGstCamera::onPreroll(_GstAppSink* sink, void* user_data)
{
	LogVerbose(LOG_GSTREAMER "LiteGstCamera -- onPreroll\n");
	return GST_FLOW_OK;
}

// onBuffer
GstFlowReturn LiteGstCamera::onBuffer(_GstAppSink* sink, void* user_data)
{
	//printf(LOG_GSTREAMER "LiteGstCamera onBuffer\n");
	
	if( !user_data )
		return GST_FLOW_OK;
		
	LiteGstCamera* dec = (LiteGstCamera*)user_data;
	
	dec->checkBuffer();
	dec->checkMsgBus();
	return GST_FLOW_OK;
}
	

#define release_return { gst_sample_unref(gstSample); return; }

// checkBuffer
void LiteGstCamera::checkBuffer()
{
	if( !mAppSink )
		return;

	// block waiting for the buffer
	GstSample* gstSample = gst_app_sink_pull_sample(mAppSink);
	
	if( !gstSample )
	{
		LogError(LOG_GSTREAMER "LiteGstCamera -- app_sink_pull_sample() returned NULL...\n");
		return;
	}
	
	// retrieve sample caps
	GstCaps* gstCaps = gst_sample_get_caps(gstSample);
	
	if( !gstCaps )
	{
		LogError(LOG_GSTREAMER "LiteGstCamera -- gst_sample had NULL caps...\n");
		release_return;
	}
	
	// retrieve the buffer from the sample
	GstBuffer* gstBuffer = gst_sample_get_buffer(gstSample);
	
	if( !gstBuffer )
	{
		LogError(LOG_GSTREAMER "LiteGstCamera -- app_sink_pull_sample() returned NULL...\n");
		release_return;
	}

	if( !mBufferManager->Enqueue(gstBuffer, gstCaps) )
		LogError(LOG_GSTREAMER "LiteGstCamera -- failed to handle incoming buffer\n");
	
	release_return;
}


// Capture
bool LiteGstCamera::Capture( void** output, imageFormat format, uint64_t timeout )
{
	// verify the output pointer exists
	if( !output )
		return false;

	// confirm the camera is streaming
	if( !mStreaming )
	{
		if( !Open() )
			return false;
	}

	// wait until a new frame is recieved
	if( !mBufferManager->Dequeue(output, format, timeout) )
	{
		LogError(LOG_GSTREAMER "gstDecoder -- failed to retrieve next image buffer\n");
		return false;
	}
	
	return true;
}

// Open
bool LiteGstCamera::Open()
{
	if( mStreaming )
		return true;

	// transition pipline to STATE_PLAYING
	LogInfo(LOG_GSTREAMER "opening LiteGstCamera for streaming, transitioning pipeline to GST_STATE_PLAYING\n");
	
	const GstStateChangeReturn result = gst_element_set_state(mPipeline, GST_STATE_PLAYING);

	if( result != GST_STATE_CHANGE_SUCCESS && result != GST_STATE_CHANGE_ASYNC)
	{
		LogError(LOG_GSTREAMER "LiteGstCamera failed to set pipeline state to PLAYING (error %u)\n", result);
		return false;
	}

	checkMsgBus();
	usleep(100*1000);
	checkMsgBus();

	mStreaming = true;
	return true;
}
	
// Close
void LiteGstCamera::Close()
{
	if( !mStreaming )
		return;

	// stop pipeline
	LogInfo(LOG_GSTREAMER "LiteGstCamera -- stopping pipeline, transitioning to GST_STATE_NULL\n");

	const GstStateChangeReturn result = gst_element_set_state(mPipeline, GST_STATE_NULL);

	if( result != GST_STATE_CHANGE_SUCCESS )
		LogError(LOG_GSTREAMER "LiteGstCamera failed to set pipeline state to PLAYING (error %u)\n", result);

	usleep(250*1000);	
	checkMsgBus();
	mStreaming = false;
	LogInfo(LOG_GSTREAMER "LiteGstCamera -- pipeline stopped\n");
}

// checkMsgBus
void LiteGstCamera::checkMsgBus()
{
	while(true)
	{
		GstMessage* msg = gst_bus_pop(mBus);

		if( !msg )
			break;

		gst_message_print(mBus, msg, this);
		gst_message_unref(msg);
	}
}

