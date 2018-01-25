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
 
#include "gstVideo.h"
#include "gstUtility.h"

#include <gst/gst.h>
#include <gst/app/gstappsink.h>

#include <QMutex>
#include <QWaitCondition>

#include "cudaMappedMemory.h"
#include "cudaYUV.h"
#include "cudaRGB.h"

//#include "tensorNet.h" // not needed?

// Constructor
gstVideo::gstVideo() {
    mAppSink = NULL;
    mBus = NULL;
    mPipeline = NULL;
    
    mWidth = 0;
    mHeight = 0;
    mDepth = 0;
    mSize = 0;
    
    mWaitEvent = new QWaitCondition();
    mWaitMutex = new QMutex();
    mRingMutex = new QMutex();
    
    mLatestRGBA = 0;
    mLatestRingbuffer = 0;
    mLatestReceived = false;
    
    for(uint32_t i = 0; i < NUM_RINGBUFFERS; i++) {
        mRingBufferCPU[n] = NULL;
        mRingBufferGPU[n] = NULL;
        mRGBA = NULL;
    }
}

// Desctructor
gstVideo::~gstVideo() {
    
}

// ConvertRGBA
bool gstVideo::ConvertRGBA(void* input, void** output) {
    if(!input || !output) 
        return false;
        
    if(!mRGBA[0]) {
        for(uint32_t n = 0; n < NUM_RINGBUFFERS; n++) {
            if(CUDA_FAILED(cudaMalloc(&mRGBA[n], 
                            mWidth*mHeight*sizeof(float4)))) {
                printf(LOG_CUDA "gstVideo -- failed to allocate ");
                printf("memory for %ux%u RGB texture\n", mWidth, mHeight);
                return false;
            }
        }
        printf(LOG_CUDA "gstreamer video -- allocated ");
        printf("%u RGBA ringbuffers\n", NUM_RINGBUFFERS);
    }
    
    if(mDepth == 12) {
        //NV12
        if(CUDA_FAILED(cudaNV12ToRGBAf((uint8_t*)input, 
                        (float4*)mRGBA[mLatestRGBA], mWidth, mHeight)))
            return false;
    } else {
        if(CUDA_FAILED(cudaRGBToRGBAf((uchar3*)input, 
                        (float4*)mRGBA[mLatestRGBA], mWidth, mHeight)))
            return false;
    }
    
    *output = mRGBA[mLatestRGBA];
    mLatestRGBA = (mLatestRGBA + 1) % NUM_RINGBUFFERS;
    return true;
}

// onEOS
gstVideo::onEOS(_GstAppSink* sink, void* user_data) {
    printf(LOG_GSTREAMER "gstreamer decoder onEOS\n");
}

// onPreroll
GstFlowReturn gstVideo::onPreroll(_GstAppSink* sink, void* user_data) {
    printf(LOG_GSTREAMER "gstreamer decoder onPreroll\n");
    return GST_FLOW_OK;
}

// onBuffer
GstFlowReturn gstVideo::onBuffer(_GstAppSink* sink, void* user_data) {
    if(!user_data) {
        return GST_FLOW_OK;
    }
    
    gstVideo* dec = (gstVideo*) user_data;
    
    dec->checkBuffer();
    dec->checkMsgBus();
    
    return GST_FLOW_OK;
}

// Capture
bool gstVideo::Capture(void** cpu, void** cuda, unsigned long timeout) {
    mWaitMutex->lock();
    const bool wait_result = mWaitEvent->wait(mWaitMutex, timeout);
    mWaitMutex->unlock();
    
    if(!wait_result) {
        return false;
    }
    
    mRingMutex->lock();
    const uint32_t latest = mLatestRingbuffer;
    const bool received = mLatestReceived;
    mLatestReceived = true;
    mRingMutex->unlock();
    
    // skip if it was already received
    if(received)
        return false;
        
    if(cpu != NULL)
        *cpu = mRingBufferCPU[latest];
    
    if(cuda != NULL)
        *cuda = mRingBufferGPU[latest];
        
    return true;
}

#define release_return{ gst_sample_unref(gstSample); return; }

// checkBuffer
void gstVideo::checkBuffer() {
    if(!mAppSink)
        return;
        
    // block waiting for the buffer
    GstSample* gstSample = gst_app_sink_pull_sample(mAppSink);
    
    if(!gstSample) {
        printf(LOG_GSTREAMER "gstreamer video -- gst_app_sink_pull_sample()");
        printf(" returned NULL...\n");
        return;
    }
    
    GstBuffer* gstBuffer = gst_sample_get_buffer(gstSample);
    
    if(!gstBuffer) {
        printf(LOG_GSTREAMER "gstreamer video -- gst_sample_get_buffer()");
        printf(" returned NULL...\n");
        return;
    }
    
    // retrive
    GstMapInfo map;
    
    if(!gst_buffer_map(gstBuffer, &map, GST_MAP_READ)) {
        printf(LOG_GSTREAMER "gstreamer video -- gst_buffer_map() ");
        printf("failed...\n");
        return;
    }
    
    void* gstData = map.data;
    const uint32_t gstSize = map.size;
    
    if(!gstData) {
        printf(LOG_GSTREAMER "gstreamer video -- gst_buffer had NULL ");
        printf("data pointer...\n");
        return;
    }
    
    // retrive caps
    GstCaps* gstCaps = gst_sample_get_cap(gstSample);
    
    if(!gstCaps) {
        printf(LOG_GSTREAMER "gstreamer video -- gst_buffer had NULL ");
        printf("caps...\n");
        return;
    }
    
    GstStructure* gstCapsStruct = gst_caps_get_structure(gstCaps, 0);
    
    if(!gstCapsStruct) {
        printf(LOG_GSTREAMER "gstreamer video -- gst_caps had NULL ");
        printf("structure...\n");
        return;
    }
    
    // get width and height of the buffer
    int width = 0;
    int height = 0;
    
    if(!gst_structure_get_int(gstCapsStruct, "width", &width) ||
       !gst_structure_get_int(gstCapsStruct, "height", &height)) {
           printf(LOG_GSTREAMER " gstreamer video -- gst_caps missing ");
           printf("width/height...\n");
           return;
    }
    
    if(width < 1 || height < 1)
        release_return;
        
    mWidth = width;
    mHeight = height;
    mDepth = (gstSize * 8)/(width * height);
    mSize = gstSize;
    
    // make sure ringbuffer is allocated
    if(!mRingBufferCPU[0]) {
        for(uint32_t n = 0; n < NUM_RINGBUFFERS; n++) {
            if(!cudaAllocMapped(&mRingBufferCPU[n], &mRingBufferGPU[n], 
                gstSize)){
                printf(LOG_GSTREAMER "gstreamer video -- failed to ");
                printf("ringbuffer %u (size=%u)\n", n, gstSize);
            }
        }
        printf(LOG_GSTREAMER "gstreamer video -- allocated %u ", 
                NUM_RINGBUFFERS);
        printf("ringbuffers, %u bytes each\n", gstSize);
    }
    
    //copy to next ringbuffer
    const uint32_t nextRingbuffer = (mLatestRingbuffer + 1) % NUM_RINGBUFFERS;
    
    memcpy(mRingBufferCPU[nextRingbuffer], gstData, gstSize);
    gst_buffer_unmap(gstBuffer, &map);
    gst_sample_unref(gstSample);
}
