#ifndef DOL_MVM_MVM_H
#define DOL_MVM_MVM_H

#include <dol_api/mvm/MVMHandle.h>
#include <dol_api/mvm/MVMProcessing.h>

namespace dol
{
    template<typename T, int ThreadsNb, int PL, int VL>
    class MVM
    {
        using Handle = MVMHandle<T, PL>;
        using Processing = MVMProcessing<T, ThreadsNb, PL, VL>;

        Handle handle;
        Processing processing;

    public:
        CI_DEVICE MVM(T * shared, int x, int y, int xAlign):
            handle(shared), processing(x, y, xAlign)
        {}

        CI_DEVICE void compute(const T * A, const T * X, T * Y) {
            processing.compute(handle, A, X, Y);
        }
    };
    
}

#endif //DOL_MVM_MVM_H