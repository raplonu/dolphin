#ifndef DOL_MVM_MVM_HANDLE_H
#define DOL_MVM_MVM_HANDLE_H

#include <dol_api/function.h>
#include <dol_api/reduce.h>
#include <dol_api/VectorType.h>

namespace dol
{
    template<typename T, int PL = 1>
    struct MVMHandle
    {
        using VectType = VectorType<T, PL>;

        VectType * reduceVect;

        __device__ MVMHandle(T * shared)//: reduceVect(shared)
        {
            auto data = (unsigned char*)shared;
            reduceVect = (VectType *)nextMul((int64_t) data, (int64_t)128);
        }

        __device__ void reduce(VectType & data)
        {
            data = blockReduce(data, reduceVect);
        }
    };
}

#endif //DOL_MVM_MVM_HANDLE_H