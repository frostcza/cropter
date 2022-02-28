#include <jetson-utils/cudaMappedMemory.h>
#include "LiteRingBuffer.h"

LiteRingBuffer::LiteRingBuffer(uint32_t n, size_t nBytes):  mNumBuf(n), mBytesBuf(nBytes)
{
	mLastestRead = 0;
	mLastestWrite = 0;
	mBuffers = (void**)malloc(n*sizeof(void*));
	mBuffTable.clear();
	for(int ii=0; ii<n; ii++)
	{
		cudaAllocMapped(&mBuffers[ii], nBytes);
		mBuffTable.insert(std::pair<void*, int>(mBuffers[ii], ii));
	}

	mMutexs = new RWMutex[n];
}

LiteRingBuffer::~LiteRingBuffer()
{
	delete []mMutexs;
	for(int ii=0; ii<mNumBuf; ii++)
	{
		cudaFreeHost(mBuffers[ii]);
	}
	free(mBuffers);
}

bool LiteRingBuffer::WriteBunk(void* src, uint32_t wBytes)
{
	if(!src)
		return false;

	// wBytes exceed the maximum bytes value
	if(wBytes > mBytesBuf)
		wBytes = mBytesBuf;

	uint32_t current = (mLastestWrite + 1) % mNumBuf;
	mMutexs[current].WLock();
	cudaMemcpy(mBuffers[current], src, wBytes, cudaMemcpyHostToHost);
	mLastestWrite = current;
	mMutexs[current].Unlock();
	return true;
}

bool LiteRingBuffer::ReadBunk(void* dst, uint32_t rBytes)
{
	if(!dst)
		return false;

	if(rBytes > mBytesBuf)
		rBytes = mBytesBuf;

	uint32_t current = mLastestWrite;
	mMutexs[current].RLock();
	cudaMemcpy(dst, mBuffers[current], rBytes, cudaMemcpyHostToHost);
	mMutexs[current].Unlock();
	return true;
}

void* LiteRingBuffer::GetReadBuffer()
{
	mLWMutex.RLock();
	uint32_t current = mLastestWrite;
	mLWMutex.Unlock();

	if(mMutexs[current].AttempRLock())
        {
		return mBuffers[current];
	}
	// theoretically never happened
	return NULL;
}

void* LiteRingBuffer::GetWriteBuffer()
{
	mLWMutex.RLock();
	uint32_t current = mLastestWrite;
	mLWMutex.Unlock();

	uint32_t loopIdx = (current + 1) % mNumBuf;
	bool res = false;

	while(loopIdx != current)
	{
		if(mMutexs[loopIdx].AttempWLock())
		{
			return mBuffers[loopIdx];
			break;
		}

		// get write lock failed, try next buffer
		loopIdx = (loopIdx + 1) % mNumBuf;
	}
	// back to current buffer, all buffers were locked
	printf("[LiteRingBuffer] No buffers available!!!!\n");
	return NULL;
}

bool LiteRingBuffer::GiveBack(void* const p, bool isWrite)
{
	if(!mBuffTable.count(p))
		return false;
	
	mMutexs[mBuffTable[p]].Unlock();

	// Write
	if(isWrite)
	{
		mLWMutex.WLock();
		mLastestWrite = mBuffTable[p];
		mLWMutex.Unlock();
	}
	return true;
}
