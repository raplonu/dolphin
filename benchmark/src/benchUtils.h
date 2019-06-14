#ifndef DOL_BENCH_UTILS_H
#define DOL_BENCH_UTILS_H

#include <ci>
#include <random>
#include <algorithm>

template<typename T = int, typename Data>
void initRandom(Data && data)
{
    std::random_device rd;
    std::uniform_real_distribution<double> dist(-5, 5);

    ci::BasicHostArray<T> tmp(ma::size(data));

    for(auto & e : tmp) e = (T)dist(rd);

    tmp.copyTo(data);
}

template<typename Data1, typename Data2>
bool isEqual(const Data1 & d1, const Data2 & d2)
{
    return std::equal(ma::begin(d1), ma::end(d1), ma::begin(d2));
}

#endif //DOL_BENCH_UTILS_H