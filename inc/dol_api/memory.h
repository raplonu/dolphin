#ifndef DOL_MEMORY_H
#define DOL_MEMORY_H

#include <ma>
#include <ci>

#include <dol_api/config.h>

namespace dol
{
    //#################
    // IMPL
    //#################
    namespace impl
    {
        using Word = int4;

        template<typename T>
        constexpr int ratio{sizeof(T)/sizeof(Word)};

        template<typename T>
        struct WordTab
        {
            Word data[ratio<T>];
        };

        template< typename T>
        inline constexpr bool IsRatioBigEnoughtV = ratio<T> >= 1;

        template< typename T>
        inline constexpr bool IsMultipleOfRatioV = ratio<T> * sizeof(Word) == sizeof(T);

        template< typename T>
        inline constexpr bool CanUseVectV = IsRatioBigEnoughtV<T> && IsMultipleOfRatioV<T>;

        template<typename T>
        constexpr void testWord()
        {
            static_assert(IsRatioBigEnoughtV<T>, "T size is too small");
            static_assert(IsMultipleOfRatioV<T>, "T size is not a multiple of word size");
        }

        template<typename T>
        __device__ void copy_vect(T * __restrict__ dst, const T * src)
        {
            // Test if T is able to be vectorized
            testWord<T>();

            auto dst_v = reinterpret_cast<WordTab<T> *>(dst);
            auto src_v = reinterpret_cast<const WordTab<T> *>(src);

            *dst_v = *src_v;
        }

        template<typename T>
        __device__ void copy_simple(T * __restrict__ dst, const T * src) { *dst = *src; }

        template<typename T, bool> struct CopyProxy;

        template<typename T> struct CopyProxy<T, true>
        { static __device__ void copy(T * dst, const T * src) { copy_vect(dst, src); } };

        template<typename T> struct CopyProxy<T, false>
        { static __device__ void copy(T * dst, const T * src) { copy_simple(dst, src); } };

        template<typename T>
        __device__ void copyImpl(T * dst, const T * src)
        {
            #if(TRY_USE_MEMORY_VECT == 1)
                CopyProxy<T, CanUseVectV<T> >::copy(dst, src);
            #else
                copy_simple(dst, src);
            #endif
        }
    }


    //#################
    // COPY
    //#################

    template<typename T>
    __device__ void copy(T * dst, const T * src) { impl::copyImpl(dst, src); }

    template<typename T>
    __device__ void copyAt(T * dst, const T * src, int id) { copy(dst + id, src + id); }

    template<typename T>
    __device__ void   warpCopy(T * dst, const T * src) { copyAt(dst, src, ci::warpPos()); }

    template<typename T>
    __device__ void  blockCopy(T * dst, const T * src) { copyAt(dst, src, TID); }

    template<typename T>
    __device__ void deviceCopy(T * dst, const T * src) { copyAt(dst, src, ID); }

    //#################
    // IMPL
    //#################

    namespace impl
    {
        template<typename T>
        __device__ T loadImpl(const T * src)
        { T res; copyImpl(&res, src); return res; }

        template<typename T>
        __device__ void storeImpl(T * dst, const T & value)
        { copyImpl(dst, &value); }
    }

    //#################
    // LOAD
    //#################

    template<typename T>
    __device__ T load(const T * src) { return impl::loadImpl(src); }

    template<typename T>
    __device__ T loadAt(const T * src, int id) { return load(src + id); }

    template<typename T>
    __device__ T   warpLoad(const T * src) { return loadAt(src, ci::warpPos()); }

    template<typename T>
    __device__ T  blockLoad(const T * src) { return loadAt(src, TID); }

    template<typename T>
    __device__ T deviceLoad(const T * src) { return loadAt(src, ID); }

    //#################
    // STORE
    //#################

    template<typename T>
    __device__ void store(T * dst, const T & value) { impl::storeImpl(dst, value); }

    template<typename T>
    __device__ void storeAt(T * dst, const T & value, int id) { store(dst + id, value); }

    template<typename T>
    __device__ void   warpStore(T * dst, const T & value) { storeAt(dst, value, ci::warpPos()); }

    template<typename T>
    __device__ void  blockStore(T * dst, const T & value) { storeAt(dst, value, TID); }

    template<typename T>
    __device__ void deviceStore(T * dst, const T & value) { storeAt(dst, value, ID); }

}

#endif //DOL_MEMORY_H