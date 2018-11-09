#ifndef DOL_REDUCE_H
#define DOL_REDUCE_H

#include <ci>
#include <dol_api/memory.h>

namespace dol
{
    template<typename T>
    __device__ T warpReduce(T data)
    { 
        #pragma unroll
        for (int offset(warpsize / 2); offset > 0; offset /= 2)
            data += ci::shflDown(data, offset);

        return data;
    }

    template<typename T>
    __device__ T blockReduce(T data, T * tmpShared)
    {
        ci::SubGroup warp(TID, warpsize);

        data = warpReduce(data);

        if(warp.pos == 0)
            storeAt(tmpShared, data, warp.id);

        ci::syncthreads();

        // maybe only select ward.id == 0 to do second reduction
        return warpReduce(loadAt(tmpShared, warp.pos));
    }
}

#endif //DOL_REDUCE_H