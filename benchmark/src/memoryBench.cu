#include <benchmark/benchmark.h>

#include <ci>

#include <thrust/copy.h>
#include <cub/cub.cuh>

#include <Dolphin>

template<int NB>
struct Data{ int data[NB]; };

using DataT = Data<4>;

using namespace benchmark;
using namespace ci;
using namespace dol;


// const int ITEM_PER_THREAD{4};

//#####################################################
// KERNEL
//#####################################################

__global__ void copy_simple(DataT * __restrict__  dst, const DataT * src, int n)
{
    #pragma unroll 32
    for(int i = ID; i < n; i += SIZE_TOT)
        dst[i] = src[i];
}

__global__ void copy_vect(DataT * __restrict__  dst, const DataT * src, int n)
{
    #pragma unroll 32
    for(int i = ID; i < n; i += SIZE_TOT)
        dol::impl::copy_vect(dst + i, src + i);
}

__global__ void copy_cub(DataT * __restrict__  dst, const DataT * src, int n)
{
    using BlockLoadT = cub::BlockLoad<DataT, 1024, 4>;
    using BlockStore = cub::BlockStore<DataT, 1024, 4>;

    DataT data[4];

    BlockLoadT().Load(src, data);

    BlockStore().Store(dst, data);
}

__global__ void copy_cub_shared(DataT * __restrict__  dst, const DataT * src, int n)
{
    using BlockLoadT = cub::BlockLoad<DataT, 1024, 4>;
    using BlockStore = cub::BlockStore<DataT, 1024, 4>;


    __shared__ union {
        typename BlockLoadT::TempStorage       load; 
        typename BlockStore::TempStorage      store;
    } temp_storage;

    int block_offset = blockIdx.x * (1024 * 4); 
    DataT data[4];

    BlockLoadT(temp_storage.load).Load(src + block_offset, data);

    ci::syncthreads();

    BlockStore(temp_storage.store).Store(dst + block_offset, data);
}

//#####################################################
// BENCHMARK
//#####################################################

const int NB{2097152};

static void dol_copyMemCpy(State& state) {
    ci::DeviceContainer<DataT> src(NB), dst(NB);
    ci::Event beg, end;

    for (auto _ : state)
    {
        thrust::device_ptr<DataT> p_dst(dst.ptr());
        thrust::device_ptr<const DataT> p_src(src.ptr());

        beg.record();

        ci::copy(p_dst, p_src, NB);

        end.record();
        end.synchronize();

        state.SetIterationTime(ci::elapsedTime(beg, end) / 1000);
        
    }
    double totalData = state.iterations() * 2 * NB * sizeof(DataT);
    state.counters["Bandwidth"] = Counter(totalData, Counter::kIsRate);
}
BENCHMARK(dol_copyMemCpy)->UseManualTime();

static void dol_copySimple(State& state) {
    ci::DeviceContainer<DataT> src(NB), dst(NB);
    ci::Event beg, end;

    for (auto _ : state)
    {
        beg.record();

        copy_simple<<<NB/1024, 1024>>>(dst.ptr().get(), src.ptr().get(), NB);

        end.record();
        end.synchronize();

        state.SetIterationTime(ci::elapsedTime(beg, end) / 1000);
        
    }
    double totalData = state.iterations() * 2 * NB * sizeof(DataT);
    state.counters["Bandwidth"] = Counter(totalData, Counter::kIsRate);
}
BENCHMARK(dol_copySimple)->UseManualTime();

static void dol_copyVect(State& state) {
    ci::DeviceContainer<DataT> src(NB), dst(NB);
    ci::Event beg, end;

    for (auto _ : state)
    {
        beg.record();

        copy_vect<<<NB/1024, 1024>>>(dst.ptr().get(), src.ptr().get(), NB);

        end.record();
        end.synchronize();

        state.SetIterationTime(ci::elapsedTime(beg, end) / 1000);
    }
    double totalData = state.iterations() * 2 * NB * sizeof(DataT);
    state.counters["Bandwidth"] = Counter(totalData, Counter::kIsRate);
}
BENCHMARK(dol_copyVect)->UseManualTime();

void dol_copyCub(State& state) {
    ci::DeviceContainer<DataT> src(NB), dst(NB);
    ci::Event beg, end;

    for (auto _ : state)
    {
        beg.record();

        copy_cub<<<NB/1024/4, 1024>>>(dst.ptr().get(), src.ptr().get(), NB);

        end.record();
        end.synchronize();

        state.SetIterationTime(ci::elapsedTime(beg, end) / 1000);
    }
    double totalData = state.iterations() * 2 * NB * sizeof(DataT);
    state.counters["Bandwidth"] = Counter(totalData, Counter::kIsRate);

}
// Disabled because not optimized
// BENCHMARK(dol_copyCub)->UseManualTime();

void dol_copyCubShared(State& state) {
    ci::DeviceContainer<DataT> src(NB), dst(NB);
    ci::Event beg, end;

    for (auto _ : state)
    {
        beg.record();

        copy_cub_shared<<<NB/1024/4, 1024>>>(dst.ptr().get(), src.ptr().get(), NB);

        end.record();
        end.synchronize();

        state.SetIterationTime(ci::elapsedTime(beg, end) / 1000);
    }
    double totalData = state.iterations() * 2 * NB * sizeof(DataT);
    state.counters["Bandwidth"] = Counter(totalData, Counter::kIsRate);

}
// Disabled because not optimized
//BENCHMARK(dol_copyCubShared)->UseManualTime();

static void dol_copyThrust(State& state) {
    ci::DeviceContainer<DataT> src(NB), dst(NB);
    ci::Event beg, end;

    for (auto _ : state)
    {
        beg.record();

        thrust::copy(src.ptr(), src.ptr() + NB, dst.ptr());

        end.record();
        end.synchronize();

        state.SetIterationTime(ci::elapsedTime(beg, end) / 1000);
    }
    double totalData = state.iterations() * 2 * NB * sizeof(DataT);
    state.counters["Bandwidth"] = Counter(totalData, Counter::kIsRate);

}
BENCHMARK(dol_copyThrust)->UseManualTime();