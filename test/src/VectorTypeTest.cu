#include <gtest/gtest.h>

#include <ci>
#include <half/half.hpp>
#include <Dolphin>

#include "testUtils.h"

using namespace dol;
using namespace ci;

CI_START_GTEST

template<typename T>
__global__ void ker_add(VectorType<T, 4> * a, const VectorType<T, 4> * b, const VectorType<T, 4> * c)
{
    if(ID0)
        *a = *b + *c;
}

template<typename T>
__global__ void ker_fma(VectorType<T, 4> * a, const VectorType<T, 4> * b, const VectorType<T, 4> * c)
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

    TEST(VectorTypeTest, HalfAddTest)
    {
        using half_h = half_float::half;
        using half_d = half;

        using VTH = VectorType<half_h, 4>;
        using VTD = VectorType<half_d, 4>;

        DeviceContainer<VTH> a(1), b(1), c(1);

        VTH tmp;
        initRandom<half_h>(tmp.data); *(a.begin()) = tmp;
        initRandom<half_h>(tmp.data); *(b.begin()) = tmp;

        ker_add<<<1,1>>>(
            reinterpret_cast<VTD*>(a.ptr().get()), 
            reinterpret_cast<VTD*>(b.ptr().get()), 
            reinterpret_cast<VTD*>(c.ptr().get())
        );

        ci::synchronize();

        VTH ad = *a.begin();
        VTH bd = *b.begin();
        VTH cd = *c.begin();

        for(int i{}; i < 4; ++i)
        {
            float a = ad.data[i], b = bd.data[i] + cd.data[i];
            EXPECT_EQ(a, b);
        }
    }

    TEST(VectorTypeTest, HalfFMATest)
    {
        using half_h = half_float::half;
        using half_d = half;

        using VTH = VectorType<half_h, 4>;
        using VTD = VectorType<half_d, 4>;

        DeviceContainer<VTH> a(1), b(1), c(1);

        VTH tmp;
        initRandom<half_h>(tmp.data); *(c.begin()) = tmp;
        initRandom<half_h>(tmp.data); *(b.begin()) = tmp;
        initRandom<half_h>(tmp.data); *(a.begin()) = tmp;

        ker_fma<<<1,1>>>(
            reinterpret_cast<VTD*>(a.ptr().get()), 
            reinterpret_cast<VTD*>(b.ptr().get()), 
            reinterpret_cast<VTD*>(c.ptr().get())
        );

        ci::synchronize();

        VTH ad = *a.begin();
        VTH bd = *b.begin();
        VTH cd = *c.begin();

        for(int i{}; i < 4; ++i)
        {
            float a = ad.data[i], b = bd.data[i] * cd.data[i] + tmp.data[i];
            EXPECT_NEAR(a, b, 0.1f);
        }
    }


}

CI_STOP_GTEST
