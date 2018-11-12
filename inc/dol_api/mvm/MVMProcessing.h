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
    template<typename T, int ThreadsNb, int PL, int VL>
    class MVMProcessing
    {
        using MVMHandle = MVMHandle<T, PL>;
        /**
         * @brief Represent a vector of VL elements T
         * 
         */
        using VectType = VectorType<T, VL>;
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
            linePerBlock(itemPerBlockLocalNb(y)), //FIX THIS !!!
            lineStart(BID * itemPerBlockNb(y)), //FIX THIS !!!
            lineStartIrregular(lineStart + (linePerBlock / PL) * PL),
            lineRemaining(linePerBlock % PL)
        {
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
                return loadAt(A, (lineStart + y) * xAlignDim + x);
            };
        }

        __device__ auto matrixLoaderIrregular(const VectType * A) {
            return [this, A] __device__ (int y) {
                return loadAtIrregular(A, (lineStart + y) * xAlignDim + xDim);
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

        __device__ auto exporter(VectType * Y, int line) {
            return [this, Y = Y + line] __device__ (const VectType & data) {
                store(Y, data);
            };
        }

        __device__ auto exporterIrregular(VectType * Y) {
            return [this, Y = Y + lineStartIrregular] __device__ (const VectType & data) {
                auto YPtr = reinterpret_cast<T *>(Y);

                for(int i{}; i < lineRemaining; ++i)
                    storeAt(YPtr, data.get(i), i);
            };
        }

        template <typename Compute, typename Export>
        __device__ void computeBlockGeneric(MVMHandle & handle, const VectType * A, const VectType * X,
                                                    Compute com, Export exp) {
            BlockType localSpace;

            int chunk(TID);
            for(; chunk < xDim; chunk += ThreadsNb)
                com(localSpace, loadAt(X, chunk), matrixLoader(A, chunk));

            if(chunk == xDim)
                com(localSpace, loadAtIrregular(X, chunk), matrixLoaderIrregular(A));

            VectType tmp = localSpace.innerReduce();

            ci::syncthreads();

            handle.reduce(tmp);

            ci::syncthreads();

            exp(tmp);
        }

        __device__ void computeBlockRegular(MVMHandle & handle, const VectType * A, const VectType * X, VectType * Y, int line)
        {
            computeBlockGeneric(
                handle, A, X,
                computer(line),
                exporter(Y, line)
            );
        }

        __device__ void computeBlockIrregular(MVMHandle & handle, const VectType * A, const VectType * X, VectType * Y) {
            computeBlockGeneric(
                handle, A, X,
                computerIrregular(),
                exporterIrregular(Y)
            );
        }

        __device__ void computeRegular(MVMHandle & handle, const VectType * A, const VectType * X, VectType * Y) {
            for (int line(0); line < linePerBlock; line += PL)
                computeBlockRegular(handle, A, X, Y, line);
        }

        __device__ void computeIrregular(MVMHandle & handle, const VectType * A, const VectType * X, VectType * Y) {
            if (lineRemaining > 0) computeBlockIrregular(handle, A, X, Y);
        }

        __device__ void compute_(MVMHandle & handle, const VectType * A, const VectType * X, VectType * Y) {
            computeRegular(handle, A, X, Y);

            ci::syncthreads();

            computeIrregular(handle, A, X, Y);
        }

        // Compute Y = A * X
        __device__ void compute(MVMHandle & handle, const T * A, const T * X, T * Y) {
            compute_(handle,
                reinterpret_cast<const VectType*>(A),
                reinterpret_cast<const VectType*>(X),
                reinterpret_cast<VectType*>(Y));
        }

    };
}

#endif //DOL_MVM_MVM_PROCESSING_H

















        // void compute_(MVMHandle & handle, const VectType * matrix, const VectType * X, VectType * Y)
        // {
        //     for(int line{offset}; line < linePerBlock; line += PL)
        //     {
        //         // Create accumulator
        //         VectType[PL] accumulator; accumulator.fill(zero<T>());

        //         // Create storage space for loading matrix
        //         VectType[PL] matData;

        //         for(int chunk{0}; chunk < y; chunk += ThreadsNb)
        //         {
        //             loadMatrix(matData, matrix + chunk + line * yAlign)


        //             #pragma unroll
        //             for(int j{}; j < PL; ++j)
        //                 accumulator[j] += matData[j];
        //         }

        //         VectType<T, PL> tmp;

        //         #pragma unroll
        //         for(int j{}; j < PL; ++j)
        //             tmp.get(j) = accumulator[j].innerReduce();

        //         tmp = blockReduce(tmp, shared);

        //         if(TID == 0)
        //             storeAt(Y + BID, tmp);
        //     }

        //     //Process remaining lines


        // }