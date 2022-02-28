#include "ModuleTemplate.h"

ModuleTemplate::ModuleTemplate(size_t numBytes, uint32_t numBuf)
{
	mReceivedStopSignal = false;
	mThreadStarted = false;
	mEvents.clear();
	mRingBuf = new LiteRingBuffer(numBuf, numBytes);
}

ModuleTemplate::~ModuleTemplate()
{
	delete mRingBuf;
	// free all events
	for(auto it=mEventsTable.begin(); it!=mEventsTable.end(); ++it)
		delete it->second;
}

bool ModuleTemplate::Read(void* caller, void** data, uint64_t timeout)
{
	Event* tEvent = NULL;
	if(!caller || !data)
		return false;
	
	// the caller has registered a event, lookup it
	if(mEventsTable.count(caller))
	{
		mMutexEvent.Lock();
		tEvent = mEventsTable[caller];
		mMutexEvent.Unlock();
	}
	else // register a event and add to event table
	{
		tEvent = new Event();
		
		mMutexEvent.Lock();
		mEventsTable.insert(std::pair<void*, Event*>(caller, tEvent));
		mMutexEvent.Unlock();
	}

	// wait the data is ready with timeout. 每次取图时无条件等待，生产者（相机线程）每写完一帧数据，即唤醒所有正在等待的Event
	if( !tEvent->Wait(timeout) )
		return false;

	// get the buffer pointer from the ringbuffer
	void* tdata = mRingBuf->GetReadBuffer();
	if(!tdata)
		return false;

	*data = tdata;
	return true;
}

bool ModuleTemplate::Write(void** data)
{
	if(!data)
		return false;

	// get the buffer pointer from the ringbuffer
	void* tdata = mRingBuf->GetWriteBuffer();
	if(!tdata)
	{
		printf("[Template] Cannot get write buffer!!!\n");
		return false;
	}

	*data = tdata;	
	return true;
}

bool ModuleTemplate::WakeAllEvent()
{
	mMutexEvent.Lock();
	for(auto it = mEventsTable.begin(); it!=mEventsTable.end(); ++it)
		it->second->Wake();
	mMutexEvent.Unlock();
	return true;
}

bool ModuleTemplate::QuerySignal()
{
	bool t;
	mMutexRSS.Lock();
	t = mReceivedStopSignal;
	mMutexRSS.Unlock();
	return t;
}
