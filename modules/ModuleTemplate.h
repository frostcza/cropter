#ifndef __MODULE_TEMPLATE_H__
#define __MODULE_TEMPLATE_H__

#include "LiteRingBuffer.h"
#include <vector>
#include <unordered_map>
#include <jetson-utils/Mutex.h>
#include <jetson-utils/Event.h>
#include <jetson-utils/Thread.h>

/* Base class as a template module
 */
class ModuleTemplate
{
public:
	/* Constructor
	 * @param numBuf 	- the number of ringBuffers
	 * @param numBytes	- the bytes of each buffer
	 */
	ModuleTemplate(size_t numBytes, uint32_t numBuf=8);

	// Deconstructor
	~ModuleTemplate();

	bool WakeAllEvent();

	//获取读buffer的指针
	bool Read(void* mod, void** data, uint64_t timeout);

	// for module itself 获取写buffer的指针
	bool Write(void** data);

	inline bool WriteFinish(void* const p) { return mRingBuf->GiveBack(p, true); }

	inline bool ReadFinish(void* const p) { return mRingBuf->GiveBack(p, false); }

	// join to wait the thread exit
	inline void Join()	{ if(mThreadStarted) {pthread_join(*(mthread.GetThreadID()), NULL); mThreadStarted=false;}	}

	// used for thread to query the exit signal
	bool QuerySignal();

	// outside call to stop the thread
	inline void Stop()	{ mMutexRSS.Lock();	mReceivedStopSignal = true;	mMutexRSS.Unlock();	}

protected:
	LiteRingBuffer*	mRingBuf;
	Thread			mthread;

protected:
	bool 			mThreadStarted;

private:
	std::vector<Event*> mEvents;
	Mutex			mMutexEvent;
	std::unordered_map<void*, Event*> mEventsTable;

	bool			mReceivedStopSignal;
	Mutex			mMutexRSS;
};


#endif
