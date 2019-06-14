#include <gtest/gtest.h>

#include <ci>
#include <Dolphin>

#include "testUtils.h"

using namespace dol;
using namespace ci;

CI_START_GTEST

__global__ void ker_loadAt(int * dataOut, const int * dataIn, int n)
{
    if(ID0)
        for(int i{}; i < n;  ++i)
            dataOut[i] = loadAt(dataIn, i);
}

__global__ void ker_warpLoad(int * dataOut, const int * dataIn, int n)
{
    int offset = BID * SIZE_B + warpsize * ci::warpId();
    if(ID < n)
        dataOut[ID] = warpLoad(dataIn + offset);
}

__global__ void ker_blockLoad(int * dataOut, const int * dataIn, int n)
{
    int offset = BID * SIZE_B;
    if(ID < n)
        dataOut[ID] = blockLoad(dataIn + offset);
}

__global__ void ker_deviceLoad(int * dataOut, const int * dataIn, int n)
{
    if(ID < n)
        dataOut[ID] = deviceLoad(dataIn);
}

__global__ void ker_vect_load(int * dataOut, const int * dataIn, int n)
{
    if(ID0)
        for(int i{}; i < n;  ++i)
            dataOut[i] = loadAt(dataIn, i);
}

const int NB = 512;

namespace
{
    TEST(loadTest,  SimpleLoadAt)
    {
        BasicDeviceArray<int> dataIn(NB), dataOut(NB);
        initRandom(dataIn);

        ker_loadAt<<<1,1>>>(dataOut.ptr().get(), dataIn.ptr().get(), NB);

        ci::synchronize();

        EXPECT_TRUE(isEqual(dataOut, dataIn));
    }

    TEST(loadTest, WarpLoad)
    {
        BasicDeviceArray<int> dataIn(NB), dataOut(NB);
        initRandom(dataIn);

        ker_warpLoad<<<8,NB/8>>>(dataOut.ptr().get(), dataIn.ptr().get(), NB);

        ci::synchronize();

        EXPECT_TRUE(isEqual(dataOut, dataIn));
    }

    TEST(loadTest, BlockLoad)
    {
        BasicDeviceArray<int> dataIn(NB), dataOut(NB);
        initRandom(dataIn);

        ker_blockLoad<<<8,NB/8>>>(dataOut.ptr().get(), dataIn.ptr().get(), NB);

        ci::synchronize();

        EXPECT_TRUE(isEqual(dataOut, dataIn));
    }

    TEST(loadTest, DeviceLoad)
    {
        BasicDeviceArray<int> dataIn(NB), dataOut(NB);
        initRandom(dataIn);

        ker_deviceLoad<<<8,NB/8>>>(dataOut.ptr().get(), dataIn.ptr().get(), NB);

        ci::synchronize();

        EXPECT_TRUE(isEqual(dataOut, dataIn));
    }
}

CI_STOP_GTEST
