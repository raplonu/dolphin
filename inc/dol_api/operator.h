#ifndef DOL_OPERATOR_H
#define DOL_OPERATOR_H

#include <cuda_fp16.h>

#include <ciFwd>

namespace dol
{
    #if (CAN_CUDA_HALF)

    //#################################
    // HALF2 arithmetic operator
    //#################################

    // __device__ inline half2 operator*(half2 a, half2 b) noexcept { return __hmul2(a, b); }

    // __device__ inline half2 operator/(half2 a, half2 b) noexcept { return __h2div(a, b); }

    // __device__ inline half2 operator+(half2 a, half2 b) noexcept { return __hadd2(a, b); }

    // __device__ inline half2 operator-(half2 a, half2 b) noexcept { return __hsub2(a, b); }

    // __device__ inline half2 operator-(half2 a) noexcept { return __hneg2(a); }

    //#################################
    // HALF arithmetic operator
    //#################################

    // __device__ inline half operator*(half a, half b) noexcept { return __hmul(a, b); }

    // __device__ inline half operator/(half a, half b) noexcept { return __hdiv(a, b); }

    // __device__ inline half operator+(half a, half b) noexcept { return __hadd(a, b); }

    // __device__ inline half operator-(half a, half b) noexcept { return __hsub(a, b); }

    // __device__ inline half operator-(half a) noexcept { return __hneg(a); }

    #endif // CAN_CUDA_HALF

    //#################################
    // Generic arithmetic fucntion
    //#################################


    #if (CAN_CUDA_HALF)

    __device__ inline half  fma(half a, half b, half c)    { return ::__hfma(a, b, c); }

    __device__ inline half2 fma(half2 a, half2 b, half2 c) { return ::__hfma2(a, b, c); }

    #endif // CAN_CUDA_HALF

    __device__ inline float  fma(float a, float b, float c)       { return ::fmaf(a, b, c); }
    __device__ inline double fma(double a, double b, double c)       { return ::fma(a, b, c); }

    template<typename T>
    __device__ T fma(T a, T b, T c) { return a * b + c; }

    template <typename T> __device__ constexpr T zero() { return {}; }
}

#endif //DOL_OPERATOR_H