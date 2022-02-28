#ifndef __LITE_RINGBUFFER_H__
#define __LITE_RINGBUFFER_H__

#include <stdint.h>
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <unordered_map>

// ReadWrite Lock
class RWMutex
{
public:
	inline RWMutex()		{ pthread_rwlock_init(&mID, NULL); }
	inline ~RWMutex()		{ pthread_rwlock_destroy(&mID); }
	inline void RLock()		{ pthread_rwlock_rdlock(&mID); }
	inline void WLock()		{ pthread_rwlock_wrlock(&mID); }
	inline void Unlock()	{ pthread_rwlock_unlock(&mID); }
	inline bool AttempRLock()	{ return (pthread_rwlock_tryrdlock(&mID) == 0); }
	inline bool AttempWLock()	{ return (pthread_rwlock_trywrlock(&mID) == 0); }
private:
	pthread_rwlock_t mID;
};

class LiteRingBuffer
{
public:
	LiteRingBuffer(uint32_t n, size_t nBytes);

	~LiteRingBuffer();

	// wBytes <= nBytes
	bool WriteBunk(void* src, uint32_t wBytes);

	// rBytes <= nBytes
	bool ReadBunk(void* dst, uint32_t rBytes);

	// read: get the newest data buffer，获取当前可以读取的buffer地址
	void* GetReadBuffer();

	// write: get a available write buffer，获取当前可以写入的buffer地址
	void* GetWriteBuffer();

	// write or read finished, call this function to unlock
	bool GiveBack(void* const p, bool isWrite);

private:
	uint32_t 	mNumBuf;		//一次开辟的buffer数量，一个buffer存储一帧图像
	size_t		mBytesBuf;		//一个buffer的大小
	uint32_t	mLastestRead;	//目前已经读取的buffer位置
	uint32_t	mLastestWrite;	//目前已经写入的buffer位置
	RWMutex		mLWMutex;		// mLastestWrite Lock 
	void**		mBuffers;		//存放每一个buffer指针的数组
	
	RWMutex*	mMutexs;		// locks to protect each buffer
	
	std::unordered_map<void*, int> mBuffTable;//键--值：buffer地址--buffer个数索引
};

#endif
