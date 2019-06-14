#include <benchmark/benchmark.h>

#include <ci>
#include <half/half.hpp>

#include <Dolphin>

#include "../benchUtils.h"
#include "../cublasUtils.h"


using namespace benchmark;
using namespace ci;
using namespace ma;
using namespace dol;

using half_h = half_float::half;
using half_d = half;

template<typename T>
struct MVMData
{
    MultiLinearDeviceArray<T> A;
    BasicDeviceArray<T> X;
    BasicDeviceArray<T> Y;

    int yDim, xDim;

    MVMData(int y, int x):
        A({y, align(x)}), X(x), Y(y, T{}), yDim(y), xDim(x)
    {
        initRandom<T>(A.at(L(y), L(x)));
        initRandom<T>(X);
    }
    
};

template<>
struct MVMData<half_h>
{
    MultiLinearDeviceArray<half_h> A;
    BasicDeviceArray<half_h> X;
    BasicDeviceArray<half_h> Y;

    int yDim, xDim;

    MVMData(int y, int x):
        A({y, align(x)}), X(x), Y(y), yDim(y), xDim(x)
    {
        initRandom<half_h>(A.at(L(y), L(x)));
        initRandom<half_h>(X);
        ci::fill(Y.ptr(), half_h{}, y);
    }
};

//#####################################################
// KERNEL
//#####################################################

template<int ThreadsNb, int PL, int VL, typename T>
__global__ __launch_bounds__(1024, 2) void do_mvm(SharedReference<T> ref, int x, int y, int xAlign, const T * A, const T * X, T * Y)
{
    MVM<T, ThreadsNb, PL, VL> mvm(ref.get(), x, y, xAlign);

    mvm.compute(A, X, Y);
}

//#####################################################
// BENCHMARK
//#####################################################

// const int Y(5316), X(12800);
const int Y(16384), X(16384);

template<typename T, int PL, int VL>
void dol_MVM(State& state) {
    const int y(Y), x(X), ThreadsNb(1024), BLOCK(160);
    MVMData<T> data(y,x);
    SharedContext sc(ci::Device(0), ThreadsNb);
    auto ref = sc.registerBuffer<T>(32 * PL * 2); //x2 to be sure

    
    ci::Event beg, end;

    for (auto _ : state)
    {
        beg.record();

        do_mvm<ThreadsNb, PL, VL><<<BLOCK,ThreadsNb, sc.memoryUsed()>>> (
            ref, x, y, align(x),
            data.A.ptr().get(), data.X.ptr().get(), data.Y.ptr().get()
        );

        end.record();
        end.synchronize();

        state.SetIterationTime(ci::elapsedTime(beg, end) / 1000);
        
    }
    double totalData = state.iterations() * x * y * sizeof(T);
    state.counters["Bandwidth"] = Counter(totalData, Counter::kIsRate);
}

template<int PL, int VL>
void dol_MVM_half(State& state) {
    const int y(Y), x(X), ThreadsNb(1024), BLOCK(160);
    MVMData<half_h> data(y,x);
    SharedContext sc(ci::Device(0), ThreadsNb);
    auto ref = sc.registerBuffer<half_d>(32 * PL * 2); //x2 to be sure

    
    ci::Event beg, end;

    for (auto _ : state)
    {
        beg.record();

        do_mvm<ThreadsNb, PL, VL><<<BLOCK,ThreadsNb, sc.memoryUsed()>>>(ref, x, y, align(x), 
        reinterpret_cast<half_d*>(data.A.ptr().get()), 
        reinterpret_cast<half_d*>(data.X.ptr().get()), 
        reinterpret_cast<half_d*>(data.Y.ptr().get()));

        end.record();
        end.synchronize();

        state.SetIterationTime(ci::elapsedTime(beg, end) / 1000);
        
    }
    double totalData = state.iterations() * x * y * sizeof(half_h);
    state.counters["Bandwidth"] = Counter(totalData, Counter::kIsRate);
}

void dol_MVM_cublas(State& state) {
    const int y(Y), x(X);
    MVMData<float> data(y,x);
    cublasHandle_t handle;
    cublasCheckError(cublasCreate(&handle));
    float alf=1.0, beta=1.0;
    
    ci::Event beg, end;

    for (auto _ : state)
    {
        beg.record();

        cublasCheckError(cublasSgemv(handle, CUBLAS_OP_T,
            y, x, &alf, data.A.ptr().get(), y,
            data.X.ptr().get(), 1, &beta, data.Y.ptr().get(), 1
        ));

        end.record();
        end.synchronize();

        state.SetIterationTime(ci::elapsedTime(beg, end) / 1000);
        
    }
    double totalData = state.iterations() * x * y * sizeof(float);
    state.counters["Bandwidth"] = Counter(totalData, Counter::kIsRate);

    cublasCheckError(cublasDestroy(handle));

}


BENCHMARK_TEMPLATE(dol_MVM, float, 1, 1)->UseManualTime();
// BENCHMARK_TEMPLATE(dol_MVM, float, 1, 2)->UseManualTime();
// BENCHMARK_TEMPLATE(dol_MVM, float, 1, 4)->UseManualTime();
// BENCHMARK_TEMPLATE(dol_MVM, float, 1, 8)->UseManualTime();

// BENCHMARK_TEMPLATE(dol_MVM_half, 1, 1)->UseManualTime();
BENCHMARK_TEMPLATE(dol_MVM_half, 1, 2)->UseManualTime();
// BENCHMARK_TEMPLATE(dol_MVM_half, 1, 4)->UseManualTime();
// BENCHMARK_TEMPLATE(dol_MVM_half, 1, 8)->UseManualTime();

BENCHMARK(dol_MVM_cublas)->UseManualTime();
