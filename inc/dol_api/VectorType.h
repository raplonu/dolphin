#ifndef DOL_VECTOR_TYPE_H
#define DOL_VECTOR_TYPE_H

#include <dol_api/operator.h>

namespace dol
{
    __device__ inline auto add_l(){return [] __device__ (const auto & a, const auto & b) { return a + b; }; }
    __device__ inline auto sub_l(){return [] __device__ (const auto & a, const auto & b) { return a - b; }; }
    __device__ inline auto mul_l(){return [] __device__ (const auto & a, const auto & b) { return a * b; }; }
    __device__ inline auto div_l(){return [] __device__ (const auto & a, const auto & b) { return a / b; }; }
    
    __device__ inline auto fma_l(){return [] __device__ (const auto & a, const auto & b, const auto & c) {
        return fma(a, b, c); };
    }
    __device__ inline auto shflDown_l(int offset){return [offset] __device__ (const auto & a) {
        return ci::shflDown(a, offset); };
    }

    template<typename T, int NB> struct VectorType;

    template<typename Fn, typename T, int NB, typename... OVT>
    __host__ __device__ VectorType<T, NB> vForEach(Fn fn, const VectorType<T, NB> & vt, const OVT &... ovts);

    template<typename T, int N>
    struct VectorType
    {
        static constexpr int NB = N;
        T data[NB];

        template<int Pos> __host__ __device__ T & get() { return data[Pos]; }
        template<int Pos> __host__ __device__ const T & get() const { return data[Pos]; }

        __host__ __device__ T & get(int pos) { return data[pos]; }
        __host__ __device__ const T & get(int pos) const { return data[pos]; }

        __device__ VectorType operator+(const VectorType & ov) const { return vForEach(add_l(), *this, ov); }
        __device__ VectorType operator-(const VectorType & ov) const { return vForEach(sub_l(), *this, ov); }
        __device__ VectorType operator*(const VectorType & ov) const { return vForEach(mul_l(), *this, ov); }
        __device__ VectorType operator/(const VectorType & ov) const { return vForEach(div_l(), *this, ov); }

        __device__ VectorType& operator+=(const VectorType & ov) { *this = *this + ov; return *this;}
        __device__ VectorType& operator-=(const VectorType & ov) { *this = *this - ov; return *this;}
        __device__ VectorType& operator*=(const VectorType & ov) { *this = *this * ov; return *this;}
        __device__ VectorType& operator/=(const VectorType & ov) { *this = *this / ov; return *this;}

        __device__ VectorType fma_(const VectorType & bv, const VectorType & cv) const { return vForEach(fma_l(), *this, bv, cv); } 
        __device__ VectorType shflDown_(int offset) const { return vForEach(shflDown_l(offset), *this); } 

        __device__ VectorType& operator=(T value) { fill(value); return *this; }

        __device__ T innerReduce() const 
        {
            T res{};

            #pragma unroll
            for(int i{}; i < NB; ++i)
                res += data[i];
            
            return res;
        }

        __device__ void fill(T value)
        {
            #pragma unroll
            for(int i{}; i < NB; ++i)
                data[i] = value;
        }
    };

    template<int N>
    struct VectorType<half, N>
    {
        static constexpr int NB = N/2;
        half2 data[NB];

        template<int Pos> __host__ __device__ half & get() {
            if(Pos % 2 == 0)
                return data[Pos/2].x;
            else
                return data[Pos/2].y;
        }
        template<int Pos> __host__ __device__ const half & get() const {
            if(Pos % 2 == 0)
                return data[Pos/2].x;
            else
                return data[Pos/2].y;
        }

        __host__ __device__ half & get(int pos) {
            if(pos % 2 == 0)
                return data[pos/2].x;
            else
                return data[pos/2].y;
        }
        __host__ __device__ const half & get(int pos) const {
            if(pos % 2 == 0)
                return data[pos/2].x;
            else
                return data[pos/2].y;
        }

        __device__ VectorType operator+(const VectorType & ov) const { return vForEach(add_l(), *this, ov); }
        __device__ VectorType operator-(const VectorType & ov) const { return vForEach(sub_l(), *this, ov); }
        __device__ VectorType operator*(const VectorType & ov) const { return vForEach(mul_l(), *this, ov); }
        __device__ VectorType operator/(const VectorType & ov) const { return vForEach(div_l(), *this, ov); }

        __device__ VectorType& operator+=(const VectorType & ov) { *this = *this + ov; return *this;}
        __device__ VectorType& operator-=(const VectorType & ov) { *this = *this - ov; return *this;}
        __device__ VectorType& operator*=(const VectorType & ov) { *this = *this * ov; return *this;}
        __device__ VectorType& operator/=(const VectorType & ov) { *this = *this / ov; return *this;}

        __device__ VectorType fma_(const VectorType & bv, const VectorType & cv) const { return vForEach(fma_l(), *this, bv, cv); } 
        __device__ VectorType shflDown_(int offset) const { return vForEach(shflDown_l(offset), *this); } 

        __device__ VectorType& operator=(half value) { fill(value); return *this; }

        __device__ half innerReduce() const 
        {
            half res{};

            #pragma unroll
            for(int i{}; i < NB; ++i)
                res += (data[i].x + data[i].y);
            
            return res;
        }

        __device__ void fill(half value)
        {
            half2 tmp{value, value};

            #pragma unroll
            for(int i{}; i < NB; ++i)
                data[i] = tmp;
        }
    };


    template<>
    struct VectorType<half, 1>
    {
        static constexpr int NB = 1;
        half data;

        template<int Pos> __host__ __device__ half & get() {
            return data;
        }
        template<int Pos> __host__ __device__ const half & get() const {
            return data;
        }

        __host__ __device__ half & get(int pos) {
            return data;
        }
        __host__ __device__ const half & get(int pos) const {
            return data;
        }

        __device__ VectorType operator+(const VectorType & ov) const { return VectorType{data + ov.data}; }
        __device__ VectorType operator-(const VectorType & ov) const { return VectorType{data - ov.data}; }
        __device__ VectorType operator*(const VectorType & ov) const { return VectorType{data * ov.data}; }
        __device__ VectorType operator/(const VectorType & ov) const { return VectorType{data / ov.data}; }

        __device__ VectorType& operator+=(const VectorType & ov) { *this = *this + ov; return *this;}
        __device__ VectorType& operator-=(const VectorType & ov) { *this = *this - ov; return *this;}
        __device__ VectorType& operator*=(const VectorType & ov) { *this = *this * ov; return *this;}
        __device__ VectorType& operator/=(const VectorType & ov) { *this = *this / ov; return *this;}

        __device__ VectorType fma_(const VectorType & bv, const VectorType & cv) const { return VectorType{fma(data, bv.data, cv.data)}; } 
        __device__ VectorType shflDown_(int offset) const { return VectorType{ci::shflDown(data, offset)}; } 

        __device__ VectorType& operator=(half value) { data = value; return *this; }

        __device__ half innerReduce() const  {
            return data;
        }

        __device__ void fill(half value) {
            data = value;
        }
    };

    template<typename Fn, typename T, int NB, typename... OVT>
    __device__ VectorType<T, NB> vForEach(Fn fn, const VectorType<T, NB> & vt, const OVT &... ovts)
    {
        VectorType<T, NB> res;

        #pragma unroll
        for(int i{}; i < VectorType<T, NB>::NB; ++i)
            res.data[i] = fn(vt.data[i], ovts.data[i]...);

        return res;
    }

    template<typename T, int NB>
    __device__ VectorType<T, NB> fma(VectorType<T, NB> a, VectorType<T, NB> b, VectorType<T, NB> c) { return a.fma_(b, c); }
}

namespace ci
{
    template<typename T, int NB>
    __device__ dol::VectorType<T, NB> shflDown(dol::VectorType<T, NB> a, int offset) { return a.shflDown_(offset); }
}

#endif //DOL_VECTOR_TYPE_H