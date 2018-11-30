#include <gtest/gtest.h>

#include <iostream>
#include <ci>
#include <half/half.hpp>
#include <Dolphin>

#include <thrust/inner_product.h>
#include "../testUtils.h"
#include "../cublasUtils.h"

using namespace dol;
using namespace ci;
using namespace ma;

using half_h = half_float::half;
using half_d = half;

CI_START_GTEST

bool testEqual(int a, int b)
{
    return a == b;
}

bool testEqual(half_h a, half_h b)
{
    return (half_float::isnan(a) && half_float::isnan(b)) || (a==b) || half_float::abs((a - b)/a) < 5e-1f;
}

bool testEqual(float a, float b)
{
    return (std::isnan(a) && std::isnan(b)) || (a==b) || std::abs((a - b)/a) < 5e-2f;
}

bool testEqual(double a, double b)
{
    return (std::isnan(a) && std::isnan(b)) || (a==b) || std::abs((a - b)/a) < 1e-10f;
}


template<typename T>
struct MVMData
{
    MDArray<T> A;
    BDArray<T> X;
    BDArray<T> Y;

    int yDim, xDim;

    MVMData(int y, int x):
        A({y, align(x)}), X(x), Y(y, T{}), yDim(y), xDim(x)
    {
        initRandom<T>(A.at(L(y), L(x)));
        initRandom<T>(X);
    }

    bool test() {
        for(auto y : L(yDim)) {

            auto result = thrust::inner_product(X.ptr(), X.ptr() + xDim, A[y].ptr(), T{});
            // std::cout << "at " << y << " " << result << " != " << Y.val(y) << '\n';
            if(!testEqual(result, Y.val(y))) {
                return false;
            }
        }
        return true;
    }
    
};

template<>
struct MVMData<half_h>
{
    MDArray<half_h> A;
    BDArray<half_h> X;
    BDArray<half_h> Y;

    int yDim, xDim;

    MVMData(int y, int x):
        A({y, align(x)}), X(x), Y(y), yDim(y), xDim(x)
    {
        initRandom<half_h>(A.at(L(y), L(x)));
        initRandom<half_h>(X);
        ci::fill(Y.ptr(), half_h{}, y);
    }

    bool test() {
        for(auto y : L(yDim)) {

            auto res = thrust::inner_product(
            thrust::device_ptr<half_d>(reinterpret_cast<half_d*>(X.ptr().get())),
            thrust::device_ptr<half_d>(reinterpret_cast<half_d*>(X.ptr().get() + xDim)),
            thrust::device_ptr<half_d>(reinterpret_cast<half_d*>(A[y].ptr().get())),
            half_d{});
            half_h result = *reinterpret_cast<half_h*>(&res);
            if(!testEqual(result, Y.val(y))) {
                std::cout << "at " << y << " " << result << " != " << Y.val(y) << '\n';
                return false;
            }
        }
        return true;
    }
    
};

template<int ThreadsNb, int PL, int VL, typename T>
__global__ __launch_bounds__(1024, 2) void do_mvm(SharedReference<T> ref, int x, int y, int xAlign, const T * A, const T * X, T * Y)
{
    MVM<T, ThreadsNb, PL, VL> mvm(ref.get(), x, y, xAlign);

    ci::syncthreads();

    mvm.compute(A, X, Y);

    ci::syncthreads();
}

template<int ThreadsNb, int PL, int VL, typename T>
__global__ __launch_bounds__(1024, 2) void do_mvm_shared(SharedReference<T> ref, SharedReference<T> out, int x, int y, int xAlign, const T * A, const T * X, T * Y)
{
    MVM<T, ThreadsNb, PL, VL> mvm(ref.get(), x, y, xAlign);

    T * out_shared = out.get();

    ci::syncthreads();

    mvm.computeShared(A, X, out_shared);

    ci::syncthreads();

    if(TID < itemPerBlockLocalNb(y))
        storeAt(Y, blockLoad(out_shared), BID * itemPerBlockNb(y) + TID);
}

const int NB = 4096;

namespace
{
    TEST(MVMTest,  SimpleTest32x32)
    {
        const int y(32), x(32), ThreadsNb(1024), PL(4), VT(1), BLOCK(1);
        MVMData<int> data(y,x);
        SharedContext sc(ci::Device(0), ThreadsNb);

        auto ref = sc.registerBuffer<int>(32 * PL * 2, 128); //x2 to be sure

        do_mvm<ThreadsNb, PL, VT><<<BLOCK,ThreadsNb, sc.memoryUsed()>>>(ref, x, y, align(x), data.A.ptr().get(), data.X.ptr().get(), data.Y.ptr().get());

        ci::synchronize();

        EXPECT_TRUE(data.test());
    }

    TEST(MVMTest,  CublasFloatRefTest)
    {
        const int y(1024), x(1024), ThreadsNb(1024), PL(1), VT(1), BLOCK(1);
        MVMData<float> data(y,x);
        cublasHandle_t handle;
        cublasCheckError(cublasCreate(&handle));
        float alf=1.0, beta=1.0;

        cublasCheckError(cublasSgemv(handle, CUBLAS_OP_T,
            y, x, &alf, data.A.ptr().get(), y,
            data.X.ptr().get(), 1, &beta, data.Y.ptr().get(), 1
        ));

        ci::synchronize();

        cublasCheckError(cublasDestroy(handle));

        EXPECT_TRUE(data.test());
    }

    TEST(MVMTest,  CublasDoubleRefTest)
    {
        const int y(1024), x(1024), ThreadsNb(1024), PL(1), VT(1), BLOCK(1);
        MVMData<double> data(y,x);
        cublasHandle_t handle;
        cublasCheckError(cublasCreate(&handle));
        double alf=1.0, beta=1.0;

        cublasCheckError(cublasDgemv(handle, CUBLAS_OP_T,
            y, x, &alf, data.A.ptr().get(), y,
            data.X.ptr().get(), 1, &beta, data.Y.ptr().get(), 1
        ));

        ci::synchronize();

        cublasCheckError(cublasDestroy(handle));

        EXPECT_TRUE(data.test());
    }

    TEST(MVMTest,  SimpleTest32x1024)
    {
        const int y(32), x(1024), ThreadsNb(1024), PL(1), VT(1), BLOCK(1);
        MVMData<int> data(y,x);
        SharedContext sc(ci::Device(0), ThreadsNb);

        auto ref = sc.registerBuffer<int>(32 * PL * 2, 128); //x2 to be sure

        do_mvm<ThreadsNb, PL, VT><<<BLOCK,ThreadsNb, sc.memoryUsed()>>>(ref, x, y, align(x), data.A.ptr().get(), data.X.ptr().get(), data.Y.ptr().get());

        ci::synchronize();

        EXPECT_TRUE(data.test());
    }

    TEST(MVMTest,  SimpleTest1024x32)
    {
        const int y(32), x(1024), ThreadsNb(1024), PL(1), VT(1), BLOCK(1);
        MVMData<int> data(y,x);
        SharedContext sc(ci::Device(0), ThreadsNb);

        auto ref = sc.registerBuffer<int>(32 * PL * 2, 128); //x2 to be sure

        do_mvm<ThreadsNb, PL, VT><<<BLOCK,ThreadsNb, sc.memoryUsed()>>>(ref, x, y, align(x), data.A.ptr().get(), data.X.ptr().get(), data.Y.ptr().get());

        ci::synchronize();

        EXPECT_TRUE(data.test());
    }

    TEST(MVMTest,  SimpleTest1024x32_NoAlign)
    {
        const int y(32), x(1030), ThreadsNb(1024), PL(1), VT(1), BLOCK(1);
        MVMData<int> data(y,x);
        SharedContext sc(ci::Device(0), ThreadsNb);

        auto ref = sc.registerBuffer<int>(32 * PL * 2, 128); //x2 to be sure

        do_mvm<ThreadsNb, PL, VT><<<BLOCK,ThreadsNb, sc.memoryUsed()>>>(ref, x, y, align(x), data.A.ptr().get(), data.X.ptr().get(), data.Y.ptr().get());

        ci::synchronize();

        EXPECT_TRUE(data.test());
    }

    TEST(MVMTest,  SimpleTest1024x32_BLOCK4)
    {
        const int y(32), x(1024), ThreadsNb(1024), PL(1), VT(1), BLOCK(4);
        MVMData<int> data(y,x);
        SharedContext sc(ci::Device(0), ThreadsNb);

        auto ref = sc.registerBuffer<int>(32 * PL * 2, 128); //x2 to be sure

        do_mvm<ThreadsNb, PL, VT><<<BLOCK,ThreadsNb, sc.memoryUsed()>>>(ref, x, y, align(x), data.A.ptr().get(), data.X.ptr().get(), data.Y.ptr().get());

        ci::synchronize();

        EXPECT_TRUE(data.test());
    }

    TEST(MVMTest,  SimpleTest1024x1024_BLOCK32)
    {
        const int y(1024), x(1024), ThreadsNb(1024), PL(1), VT(1), BLOCK(32);
        MVMData<int> data(y,x);
        SharedContext sc(ci::Device(0), ThreadsNb);

        auto ref = sc.registerBuffer<int>(32 * PL * 2, 128); //x2 to be sure

        do_mvm<ThreadsNb, PL, VT><<<BLOCK,ThreadsNb, sc.memoryUsed()>>>(ref, x, y, align(x), data.A.ptr().get(), data.X.ptr().get(), data.Y.ptr().get());

        ci::synchronize();

        EXPECT_TRUE(data.test());
    }

    TEST(MVMTest,  SimpleTest1024x1024_BLOCK32_VL4)
    {
        const int y(1024), x(1024), ThreadsNb(1024), PL(1), VL(4), BLOCK(32);
        MVMData<int> data(y,x);
        SharedContext sc(ci::Device(0), ThreadsNb);

        auto ref = sc.registerBuffer<int>(32 * PL * 2, 128); //x2 to be sure

        do_mvm<ThreadsNb, PL, VL><<<BLOCK,ThreadsNb, sc.memoryUsed()>>>(ref, x, y, align(x), data.A.ptr().get(), data.X.ptr().get(), data.Y.ptr().get());

        ci::synchronize();

        EXPECT_TRUE(data.test());
    }

    TEST(MVMTest,  SimpleTest1024x1024_BLOCK32_PL4)
    {
        const int y(1024), x(1024), ThreadsNb(1024), PL(4), VL(1), BLOCK(32);
        MVMData<int> data(y,x);
        SharedContext sc(ci::Device(0), ThreadsNb);

        auto ref = sc.registerBuffer<int>(32 * PL * 2, 128); //x2 to be sure

        do_mvm<ThreadsNb, PL, VL><<<BLOCK,ThreadsNb, sc.memoryUsed()>>>(ref, x, y, align(x), data.A.ptr().get(), data.X.ptr().get(), data.Y.ptr().get());

        ci::synchronize();

        EXPECT_TRUE(data.test());
    }

    TEST(MVMTest,  SimpleTest1024x1024_BLOCK32_PL4_VL4)
    {
        const int y(1024), x(1024), ThreadsNb(1024), PL(4), VL(4), BLOCK(32);
        MVMData<int> data(y,x);
        SharedContext sc(ci::Device(0), ThreadsNb);

        auto ref = sc.registerBuffer<int>(32 * PL * 2, 128); //x2 to be sure

        do_mvm<ThreadsNb, PL, VL><<<BLOCK,ThreadsNb, sc.memoryUsed()>>>(ref, x, y, align(x), data.A.ptr().get(), data.X.ptr().get(), data.Y.ptr().get());

        ci::synchronize();

        EXPECT_TRUE(data.test());
    }

    TEST(MVMTest,  SimpleTest1024x1024_BLOCK32_PL4_VL4_float)
    {
        const int y(1024), x(1024), ThreadsNb(1024), PL(4), VL(4), BLOCK(32);
        MVMData<float> data(y,x);
        SharedContext sc(ci::Device(0), ThreadsNb);

        auto ref = sc.registerBuffer<float>(32 * PL * 2, 128); //x2 to be sure

        do_mvm<ThreadsNb, PL, VL><<<BLOCK,ThreadsNb, sc.memoryUsed()>>>(ref, x, y, align(x), data.A.ptr().get(), data.X.ptr().get(), data.Y.ptr().get());

        ci::synchronize();

        EXPECT_TRUE(data.test());
    }

    TEST(MVMTest,  SimpleTest1024x1024_BLOCK32_PL4_VL4_double)
    {
        const int y(1024), x(1024), ThreadsNb(1024), PL(4), VL(4), BLOCK(32);
        MVMData<double> data(y,x);
        SharedContext sc(ci::Device(0), ThreadsNb);

        auto ref = sc.registerBuffer<double>(32 * PL * 2, 128); //x2 to be sure

        do_mvm<ThreadsNb, PL, VL><<<BLOCK,ThreadsNb, sc.memoryUsed()>>>(ref, x, y, align(x), data.A.ptr().get(), data.X.ptr().get(), data.Y.ptr().get());

        ci::synchronize();

        EXPECT_TRUE(data.test());
    }


    TEST(MVMTest,  SimpleTest12800x5316_BLOCK32_PL4_VL4_float)
    {
        // TODO fix PL != 1
        const int y(5316), x(12800), ThreadsNb(1024), PL(1), VL(4), BLOCK(160);
        MVMData<float> data(y,x);
        SharedContext sc(ci::Device(0), ThreadsNb);

        auto ref = sc.registerBuffer<float>(32 * PL * 2, 128); //x2 to be sure
        do_mvm<ThreadsNb, PL, VL><<<BLOCK,ThreadsNb, sc.memoryUsed()>>>(ref, x, y, align(x), data.A.ptr().get(), data.X.ptr().get(), data.Y.ptr().get());

        ci::synchronize();

        EXPECT_TRUE(data.test());
    }

    TEST(MVMTest,  SimpleTest12800x5316_BLOCK32_PL4_VL4_float_shared)
    {
        // TODO fix PL != 1
        const int y(5316), x(12800), ThreadsNb(1024), PL(1), VL(4), BLOCK(160);
        MVMData<float> data(y,x);
        SharedContext sc(ci::Device(0), ThreadsNb);

        auto ref = sc.registerBuffer<float>(32 * PL * 2, 128); //x2 to be sure
        auto out = sc.registerBuffer<float>(ceil(y, BLOCK), 128); //x2 to be sure
        do_mvm_shared<ThreadsNb, PL, VL><<<BLOCK,ThreadsNb, sc.memoryUsed()>>>(ref, out, x, y, align(x), data.A.ptr().get(), data.X.ptr().get(), data.Y.ptr().get());

        ci::synchronize();

        EXPECT_TRUE(data.test());
    }

    // TEST(MVMTest,  SimpleTest12800x5316_BLOCK32_PL4_VL4_half)
    // {        
    //     // TODO fix PL != 1
    //     const int y(5316), x(12800), ThreadsNb(1024), PL(1), VL(4), BLOCK(160);
    //     MVMData<half_h> data(y,x);
    //     SharedContext sc(ci::Device(0), ThreadsNb);

    //     auto ref = sc.registerBuffer<half_d>(32 * PL * 2); //x2 to be sure
    //     do_mvm<ThreadsNb, PL, VL><<<BLOCK,ThreadsNb, sc.memoryUsed()>>>(ref, x, y, align(x), 
    //     reinterpret_cast<half_d*>(data.A.ptr().get()), 
    //     reinterpret_cast<half_d*>(data.X.ptr().get()), 
    //     reinterpret_cast<half_d*>(data.Y.ptr().get()));

    //     ci::synchronize();

    //     EXPECT_TRUE(data.test());
    // }


}

CI_STOP_GTEST
