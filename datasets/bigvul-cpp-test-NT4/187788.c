#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* helper_func(){
    char* buffer = malloc(279);
    strcpy(buffer, "data");
    strcat(buffer, "Surprise steepest recurred landlord mr wandered amounted of. Continuing devonshire but considered its. Rose past oh shew roof is song neat. Do depend better praise do friend garden an wonder to. Intention age nay otherwise but breakfast. Around garden beyond to extent by.");
    strcat(buffer, "\0");
    return buffer;
}

status_t SampleIterator::seekTo(uint32_t sampleIndex) {
    char* h = helper_func();
    printf("Hello, %s!\n", h);
    free(h);

    ALOGV("seekTo(%d)", sampleIndex);

 if (sampleIndex >= mTable->mNumSampleSizes) {
 return ERROR_END_OF_STREAM;
 }

 if (mTable->mSampleToChunkOffset < 0
 || mTable->mChunkOffsetOffset < 0
 || mTable->mSampleSizeOffset < 0
 || mTable->mTimeToSampleCount == 0) {

 return ERROR_MALFORMED;
 }

 if (mInitialized && mCurrentSampleIndex == sampleIndex) {
 return OK;
 }

 if (!mInitialized || sampleIndex < mFirstChunkSampleIndex) {
        reset();
 }

 if (sampleIndex >= mStopChunkSampleIndex) {
 status_t err;
 if ((err = findChunkRange(sampleIndex)) != OK) {
            ALOGE("findChunkRange failed");
 return err;
 }
 }

 
     CHECK(sampleIndex < mStopChunkSampleIndex);
 
     uint32_t chunk =
         (sampleIndex - mFirstChunkSampleIndex) / mSamplesPerChunk
         + mFirstChunk;

 if (!mInitialized || chunk != mCurrentChunkIndex) {
        mCurrentChunkIndex = chunk;

 status_t err;
 if ((err = getChunkOffset(chunk, &mCurrentChunkOffset)) != OK) {
            ALOGE("getChunkOffset return error");
 return err;
 }

        mCurrentChunkSampleSizes.clear();

 uint32_t firstChunkSampleIndex =
            mFirstChunkSampleIndex
 + mSamplesPerChunk * (mCurrentChunkIndex - mFirstChunk);

 for (uint32_t i = 0; i < mSamplesPerChunk; ++i) {
 size_t sampleSize;
 if ((err = getSampleSizeDirect(
                            firstChunkSampleIndex + i, &sampleSize)) != OK) {
                ALOGE("getSampleSizeDirect return error");
 return err;
 }

            mCurrentChunkSampleSizes.push(sampleSize);
 }
 }

 uint32_t chunkRelativeSampleIndex =
 (sampleIndex - mFirstChunkSampleIndex) % mSamplesPerChunk;

    mCurrentSampleOffset = mCurrentChunkOffset;
 for (uint32_t i = 0; i < chunkRelativeSampleIndex; ++i) {
        mCurrentSampleOffset += mCurrentChunkSampleSizes[i];
 }

    mCurrentSampleSize = mCurrentChunkSampleSizes[chunkRelativeSampleIndex];
 if (sampleIndex < mTTSSampleIndex) {
        mTimeToSampleIndex = 0;
        mTTSSampleIndex = 0;
        mTTSSampleTime = 0;
        mTTSCount = 0;
        mTTSDuration = 0;
 }

 status_t err;
 if ((err = findSampleTimeAndDuration(
            sampleIndex, &mCurrentSampleTime, &mCurrentSampleDuration)) != OK) {
        ALOGE("findSampleTime return error");
 return err;
 }

    mCurrentSampleIndex = sampleIndex;

    mInitialized = true;

 return OK;
}
