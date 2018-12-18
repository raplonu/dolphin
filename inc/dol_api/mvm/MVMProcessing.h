#ifndef DOL_MVM_MVM_PROCESSING_H
#define DOL_MVM_MVM_PROCESSING_H

#include <dol_api/function.h>
#include <dol_api/reduce.h>
#include <dol_api/memory.h>
#include <dol_api/VectorType.h>
#include <dol_api/mvm/MVMProcessing.h>

namespace dol
{

    /**
     * @brief 
     * 
     * @tparam T 
     * @tparam PL Parallel Line aka number of line processed in parallel
     * @tparam VL Vector Length aka number of contigious element that is processed by each thread
     */
    template<typename T, int ThreadsNb, int PL = 1, int VL = 1>
    class MVMProcessing
    {
        using MVMHandle = MVMHandle<T, PL>;
        /**
         * @brief Represent a vector of PL elements T
         * 
         */
        using VectType = VectorType<T, VL>;
        using VectType2 = VectorType<T, PL>;
        /**
         * @brief Represent a vector of PL vector
         * 
         */
        using BlockType = VectorType<VectType, PL>;

        int xDim;
        int yDim;
        int xRemaining;
        int xAlignDim;

        int linePerBlock;
        int lineStart;
        int lineStartIrregular;
        int lineRemaining;

    public:

        __device__ MVMProcessing(int x, int y, int xAlign):
            xDim(x / VL), yDim(y), xRemaining(x % VL), xAlignDim(xAlign / VL), 
            linePerBlock(itemPerBlockLocalNb(y) / PL * PL),
            lineStart(BID * itemPerBlockNb(y)),
            lineStartIrregular(linePerBlock),
            lineRemaining(itemPerBlockLocalNb(y) % PL)
        {
            // if(TID0)
            // printf("[%d] xDim %d yDim %d xRemaining %d xAlignDim %d linePerBlock %d lineStart %d lineStartIrregular %d lineRemaining %d\n",
            // BID, xDim, yDim, xRemaining, xAlignDim, linePerBlock, lineStart, lineStartIrregular, lineRemaining);
            // TODO assert xAlign % VL == 0 !!
        }

        __device__ VectType loadAtIrregular(const VectType * data, int offset) {
            auto dataPtr = reinterpret_cast<const T *>(data);

            VectType res;

            int i{};
            for(; i < xRemaining; ++i)
                res.get(i) = loadAt(dataPtr, offset * VL + i);
            for(; i < VL; ++i)
                res.get(i) = T{};

            return res;
        }

        __device__ auto matrixLoader(const VectType * A, int x) {
            return [this, A, x] __device__ (int y) {
                return loadAt(A, y * xAlignDim + x);
            };
        }

        __device__ auto matrixLoaderIrregular(const VectType * A) {
            return [this, A] __device__ (int y) {
                return loadAtIrregular(A, y * xAlignDim + xDim);
            };
        }

        __device__ auto computer(int line) {
            return [this, line] __device__ (BlockType & accumulator, VectType X, auto loader) {
                #pragma unroll
                for (int i(0); i < PL; ++i)
                    accumulator.get(i) = dol::fma(loader(line + i), X, accumulator.get(i));
            };
        }

        __device__ auto computerIrregular() {
            return [this] __device__ (BlockType & accumulator, VectType X, auto loader) {
                for (int i(0); i < lineRemaining; ++i)
                    accumulator.get(i) = dol::fma(loader(lineStartIrregular + i), X, accumulator.get(i));
            };
        }

        __device__ auto exporter(VectType2 * Y) {
            return [this, Y] __device__ (const VectType2 & data) {
                if(TID0)
                // {
                //     printf("Reg %d -> %f %f\n", BID, data.get(0), data.get(1));
                    store(Y, data);
                // }
            };
        }

        __device__ auto exporterIrregular(VectType2 * Y) {
            return [this, Y] __device__ (const VectType2 & data) {
                auto YPtr = reinterpret_cast<T *>(Y);

                if(TID0)
                    for(int i{}; i < lineRemaining; ++i)
                    // {
                    //     printf("Ire %d -> %f\n", BID, data.get(i));
                        storeAt(YPtr, data.get(i), i);
                    // }
            };
        }

        template <typename Compute, typename Export>
        __device__ void computeBlockGeneric(MVMHandle & handle, const VectType * A, const VectType * X,
                                                    Compute com, Export exp) {
            
            BlockType localSpace;
            localSpace = (VectType() = 0);

            int chunk(TID);
            for(; chunk < xDim; chunk += ThreadsNb)
                com(localSpace, loadAt(X, chunk), matrixLoader(A, chunk));

            if(xRemaining && chunk == xDim)
                com(localSpace, loadAtIrregular(X, chunk), matrixLoaderIrregular(A));

            VectType2 tmp;

            #pragma unroll
            for(int i{}; i < PL; ++i)
                tmp.get(i) = localSpace.get(i).innerReduce();

            handle.reduce(tmp);

            exp(tmp);
        }

        __device__ void computeBlockRegular(MVMHandle & handle, const VectType * A, const VectType * X, VectType2 * Y, int line) {
            computeBlockGeneric(
                handle, A, X,
                computer(line),
                exporter(Y + (line / PL))
            );
        }

        __device__ void computeBlockIrregular(MVMHandle & handle, const VectType * A, const VectType * X, VectType2 * Y) {
            computeBlockGeneric(
                handle, A, X,
                computerIrregular(),
                exporterIrregular(Y + (lineStartIrregular / PL))
            );
        }

        __device__ void computeRegular(MVMHandle & handle, const VectType * A, const VectType * X, VectType2 * Y) {
            for (int line(0); line < linePerBlock; line += PL)
                computeBlockRegular(handle, A, X, Y, line);
        }

        __device__ void computeIrregular(MVMHandle & handle, const VectType * A, const VectType * X, VectType2 * Y) {
            if (lineRemaining > 0) computeBlockIrregular(handle, A, X, Y);
        }

        __device__ void compute_(MVMHandle & handle, const VectType * A, const VectType * X, VectType2 * Y) {
            computeRegular(handle, A, X, Y);

            computeIrregular(handle, A, X, Y);
        }

        /**
         * @brief Compute mvmv : Y = A * X
         * 
         * @param handle shared memory handler for reduction
         * @param A input global matrix
         * @param X input global vector
         * @param Y output GLOBAL vector
         */
        __device__ void compute(MVMHandle & handle, const T * A, const T * X, T * Y) {
            compute_(handle,
                reinterpret_cast<const VectType*>(A) + lineStart * xAlignDim,
                reinterpret_cast<const VectType*>(X),
                reinterpret_cast<VectType2*>(Y + lineStart));
        }


        /**
         * @brief Compute mvmv : Y = A * X
         * 
         * @param handle shared memory handler for reduction
         * @param A input global matrix
         * @param X input global vector
         * @param Y output SHARED vector
         */
        __device__ void computeShared(MVMHandle & handle, const T * A, const T * X, T * Y) {
            compute_(handle,
                reinterpret_cast<const VectType*>(A) + lineStart * xAlignDim,
                reinterpret_cast<const VectType*>(X),
                reinterpret_cast<VectType2*>(Y));
        }
    };
}

#endif //DOL_MVM_MVM_PROCESSING_H
