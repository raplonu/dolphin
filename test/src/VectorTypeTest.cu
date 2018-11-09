#include <gtest/gtest.h>

#include <ci>
#include <Dolphin>

#include "testUtils.h"

using namespace dol;
using namespace ci;

CI_START_GTEST

__global__ void ker_add(VectorType<int, 4> * a, const VectorType<int, 4> * b, const VectorType<int, 4> * c)
{
    if(ID0)
        *a = *b + *c;
}

__global__ void ker_fma(VectorType<int, 4> * a, const VectorType<int, 4> * b, const VectorType<int, 4> * c)
{
    if(ID0)
        *a = dol::fma(*b, *c, *a);
}

namespace
{
    TEST(VectorTypeTest, AddTest)
    {
        DeviceContainer<VectorType<int, 4>> a(1), b(1), c(1);

        VectorType<int, 4> tmp;
        initRandom(tmp.data); *(b.begin()) = tmp;
        initRandom(tmp.data); *(c.begin()) = tmp;

        ker_add<<<1,1>>>(a.ptr().get(), b.ptr().get(), c.ptr().get());

        ci::synchronize();

        VectorType<int, 4> ad = *a.begin();
        VectorType<int, 4> bd = *b.begin();
        VectorType<int, 4> cd = *c.begin();

        for(int i{}; i < 4; ++i)
            EXPECT_EQ(ad.data[i], bd.data[i] + cd.data[i]);
    }

    TEST(VectorTypeTest, FMATest)
    {
        DeviceContainer<VectorType<int, 4>> a(1), b(1), c(1);

        VectorType<int, 4> tmp;
        initRandom(tmp.data); *(c.begin()) = tmp;
        initRandom(tmp.data); *(b.begin()) = tmp;
        initRandom(tmp.data); *(a.begin()) = tmp;

        ker_fma<<<1,1>>>(a.ptr().get(), b.ptr().get(), c.ptr().get());

        ci::synchronize();

        VectorType<int, 4> ad = *a.begin();
        VectorType<int, 4> bd = *b.begin();
        VectorType<int, 4> cd = *c.begin();

        for(int i{}; i < 4; ++i)
            EXPECT_EQ(ad.data[i], bd.data[i] * cd.data[i] + tmp.data[i]);
    }


}

CI_STOP_GTEST
