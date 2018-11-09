#ifndef DOL_COPY_DEVICE_COPY_H
#define DOL_COPY_DEVICE_COPY_H

#include <ci>

namespace dol
{
    template<typename T>
    void copy(const T * in, T * __restrict__ out)
    {
        
    }

    template<typename InputIt, typename OutputIt>
    CI_HODE void localCopy(InputIt first, InputIt last, OutputIt d_first)
    {
        while(first != last)
            *(first++) = *(d_first++);
    }

    template<typename InputIt, typename OutputIt>
    CI_HODE void blockCopy(InputIt first, InputIt last, OutputIt d_first)
    {
        first += TID;
        while(first < last)
        {
            *first = *d_first;
            first += SIZE_B; d_first += SIZE_B; 
        }
    }

    template<typename InputIt, typename OutputIt>
    CI_HODE void blockCopy(InputIt first, InputIt last, OutputIt d_first)
    {
        first += ID;
        while(first < last)
        {
            *first = *d_first;
            first += SIZE_G; d_first += SIZE_G; 
        }
    }
}

#endif //DOL_COPY_DEVICE_COPY_H