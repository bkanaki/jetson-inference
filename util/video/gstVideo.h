/**
 * This file is inspired by the original work by Nvidia author dusty-nv
 * and the work by omaralvarez for RTSP Pipeline. Their codes are available 
 * on their respective GitHub profiles under the repo jetson-inference.
 * The objective of this file is to make a pipeline that is generic to video 
 * or stream by defining the gst-pipeline as one of the commandline inputs.
 * The original Nvidia license information is retained as is.
 * Author: Bhargav Kanakiya
 * email: bhargav@automot.us
 */

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
 
#ifndef __GSTREAMER_VIDEO_H__
#define __GSTREAMER_VIDEO_H__

#include <gst/gst.h>
#include <string>

struct _GstAppSink;
class QWaitCondition;
class QMutex;


/**
 * gstreamer video pipeline using gst-launch
 */
class gstVideo {
    public:
        // Create gst pipeline
        static gstVideo* Create(std::string pipeline);
        static gstVideo* Create(std::string pipeline, uint32_t width,
                                uint32_t height, uint32_t depth);
                                
        // Destroy
        ~gstVideo();
        
        // Start/Stop the stream
        void Open();
        void Close();
        
        // Capture YUV (NV12)
        bool Capture(void** cpu, void** cuda, unsigned long timeout=ULONG_MAX);
        
        // Takes in the YUV-NV12 CUDA image and converts to float4 RGBA 
        // with pixel intensity between 0-255. Done on GPU usin CUDA
        cool ConvertRGBA(void* input, void** output);
        
        // Image dimensions
        inline uint32_t GetWidth() const        { return mWidth; }
        inline uint32_t GetHeight() const       { return mHeight; }
        inline uint32_t GetPixelDepth() const   { return mDepth; }
        inline uint32_t GetSize() const         { return mSize; }
        
        // Default resolution, unless otherwise specified during Create
        static const uint32_t DefaultWidth = 1280;
        static const uint32_t DefaultHeight = 720;
    
    private:
        static void onEOS(_GstAppSink* sink, void* user_data);
        static GstFlowReturn onPreroll(_GstAppSink* sink, void* user_data);
        static GstFlowReturn onBuffer(_GstAppSink* sink, void* user_data);
        
        gstVideo();
        
        bool init();
        void checkMsgBus();
        void checkBuffer();
        // no buildLaunchStr() because we have a member for storing it below
        
        _GstBus* mBus;
        _GstAppSink* mAppSink;
        _GstElement* mPipeline;
        
        std::string mLaunchStr;
        
        uint32_t mWidth;
        uint32_t mHeight;
        uint32_t mDepth;
        uint32_t mSize;
        
        static const uint32_t NUM_RINGBUFFERS = 16;
        
        void* mRingBufferCPU[NUM_RINGBUFFERS];
        void* mRingBufferGPU[NUM_RINGBUFFERS];
        
        QWaitCondition* mWaitEvent;
        
        QMutex* mWaitMutex;
        QMutex* mRingMutex;
        
        uint32_t mLatestRGBA;
        uint32_t mLatestRingbuffer;
        bool mLatestReceived;
        
        void* mRGBA[NUM_RINGBUFFERS];
        
        /* Not needed as no camera is used
        int   mV4L2Device;	// -1 for onboard, >=0 for V4L2 device
	
	    inline bool onboardCamera() const		{ return (mV4L2Device < 0); }
	    */
};

#endif
