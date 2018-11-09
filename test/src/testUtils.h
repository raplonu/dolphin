#ifndef DOL_TEST_UTILS_H
#define DOL_TEST_UTILS_H

#include <random>
#include <algorithm>

template<typename T = int, typename Data>
void initRandom(Data & data)
{
    std::random_device rd;
    std::uniform_int_distribution<T> dist(0, 99);

    for(auto e : data) e = dist(rd);
}

template<typename Data1, typename Data2>
bool isEqual(const Data1 & d1, const Data2 & d2)
{
    return std::equal(ma::begin(d1), ma::end(d1), ma::begin(d2));
}

#endif //DOL_TEST_UTILS_H