#include <gtest/gtest.h>

#include <ci>
#include <Dolphin>

#include "testUtils.h"

using namespace dol;
using namespace ci;

CI_START_GTEST

__global__ void ker_warpReduce(int * dataOut, const int * dataIn, int n)
{
    int res;
    if(ID < n)
        res = dataIn[ID];
    else
        res = 0;
    
    res = dol::warpReduce(res);

    if(ID < n && ci::warpPos() == 0)
        dataOut[ci::warpId() + BID * 32] = res;
}

__global__ void ker_blockReduce(int * dataOut, const int * dataIn, int n)
{
    int res;
    if(ID < n)
        res = dataIn[ID];
    else
        res = 0;

    __shared__ int tmp[32];

    res = dol::blockReduce(res, tmp);

    if(TID0)
        dataOut[BID] = res;
}

const int NB = 4096;

namespace
{
    TEST(reduceTest,  WarpReduce)
    {
        BDArray<int> dataIn(NB), dataOut(NB/32);
        initRandom(dataIn);

        ker_warpReduce<<<NB/1024,1024>>>(dataOut.ptr().get(), dataIn.ptr().get(), NB);

        ci::synchronize();

        BHArray<int> result(NB/32);

        for(int i{}; i < NB/32; ++i)
        {
            auto view = dataIn.at(i*32, (i+1)*32);
            result[i].val() = std::accumulate(ma::begin(view), ma::end(view), 0);
        }

        EXPECT_TRUE(isEqual(dataOut, result));
    }

    TEST(reduceTest,  BlockReduce)
    {
        BDArray<int> dataIn(NB), dataOut(NB/1024);
        initRandom(dataIn);

        ker_blockReduce<<<NB/1024,1024>>>(dataOut.ptr().get(), dataIn.ptr().get(), NB);

        ci::synchronize();

        BHArray<int> result(NB/1024);

        for(int i{}; i < NB/1024; ++i)
        {
            auto view = dataIn.at(i*1024, (i+1)*1024);
            result[i].val() = std::accumulate(ma::begin(view), ma::end(view), 0);
        }

        EXPECT_TRUE(isEqual(dataOut, result));
    }

}

CI_STOP_GTEST
