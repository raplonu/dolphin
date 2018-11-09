#ifndef DOL_MVM_MVM_HANDLE_H
#define DOL_MVM_MVM_HANDLE_H

#include <dol_api/reduce.h>
#include <dol_api/VectorType.h>

namespace dol
{
    template<typename T, int PL>
    struct MVMHandle
    {
        using VectType = VectorType<T, PL>;

        VectType * reduceVect;

        __device__ void reduce(VectType & data)
        {
            blockReduce(data, reduceVect);
        }
    };
}

#endif //DOL_MVM_MVM_HANDLE_H