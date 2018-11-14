#ifndef DOL_FUNCTION_H
#define DOL_FUNCTION_H

#include <ciFwd>

namespace dol
{
    /**
    * Calculate the ceil result of a / b
    * @param  a numerator
    * @param  b denominator
    * @return   ceil division of a / b
    */
    template <typename T>
    CI_HODE constexpr T ceil(T a, T b) noexcept { return (a + b - 1) / b; }

    template<typename T>
    CI_HODE constexpr T nextMul(T a, T b) { return ( (a-1) / b + 1) * b; }

    template <typename T>
    CI_HODE constexpr T align(T size) noexcept { return nextMul(size, 32); }

    template <class T>
    CI_HODE constexpr T max(T a, T b) noexcept { return (a < b) ? b : a; }

    template <class T>
    CI_HODE constexpr T min(T a, T b) noexcept { return (a < b) ? a : b; }

    CI_DEVICE inline bool isNotLastBlock() { return BID < (SIZE_G - 1); }

    /**
     * @brief Return the generalized number of item per group. This number is the same for every group.
     * The sum of the result for every block may be equal or superior of intemNb
     * Esamples :
     * itemPerGroupLocalNb(100, 10) -> 10
     * itemPerGroupLocalNb(89, 10) -> 9
     * @param itemNb 
     * @param groupSize 
     * @return int itemPerGroupNb 
     */
    CI_DEVICE inline int itemPerGroupNb(int itemNb, int groupSize) { return ceil(itemNb, groupSize); }

    /**
     * @brief Share a resource between a group. Guaranty to minimize the number of resource
     * for each group element and to handle exactly the right number of available item
     * Examples :
     * Item are perfectly shared between all groups
     * itemPerGroupLocalNb(100, [0, .., 9], 10) -> [10, .., 10]
     * The last group handle only 1 element to unsure item is not handled two times
     * itemPerGroupLocalNb(7, [0, 1, 2, 3], 4) -> [2, 2, 2, 1]
     * The last group remain idle because the is not element at all to handle
     * itemPerGroupLocalNb(5, [0, 1, 2, 3], 4) -> [2, 2, 1, 0]
     * 
     * @param itemNb Total number of item to handle
     * @param groupId The id of the current group
     * @param groupSize The number of element in the group
     * @return int itemPerGroupLocalNb 
     */
    CI_DEVICE inline int itemPerGroupLocalNb(int itemNb, int groupId, int groupSize) {
        int itemPG = itemPerGroupNb(itemNb, groupSize);

        return max(0, min(itemNb, (groupId + 1) * itemPG) - groupId * itemPG);
    }

    /**
     * @brief Return the generalized number of item per kernel blocks
     * 
     * @param itemNb 
     * @return int itemPerBlockNb 
     */
    CI_DEVICE inline int itemPerBlockNb(int itemNb) { return itemPerGroupNb(itemNb, SIZE_G); }

    /**
     * @brief Return the specific number of item per kernel blocks
     * 
     * @param itemNb 
     * @return int itemPerBlockNb 
     */
    CI_DEVICE inline int itemPerBlockLocalNb(int itemNb) { return itemPerGroupLocalNb(itemNb, BID, SIZE_G); }

    /**
     * @brief Get the position of thread considering customItemPerBlockNb item per block.
     * 2 threads in 2 group can share same position. Need to test the validity of a thread
     * 
     * @param customItemPerBlockNb 
     * @return int threadItemPosCustom 
     */
    CI_DEVICE inline int threadItemPosCustom(int customItemPerBlockNb) { return TID + BID * customItemPerBlockNb; }

    /**
     * @brief Get the position of a thread considering a total number of item to be shared trough the GPU
     * 2 threads in 2 group can share same position. Need to test the validity of a thread
     * 
     * @param itemNb 
     * @return int threadItemPos 
     */
    CI_DEVICE inline int threadItemPos(int itemNb) { return threadItemPosCustom(itemPerBlockNb(itemNb)); }

    /**
     * @brief Test if a thread have items to handle
     * 
     * @param itemNb 
     * @param itemPerBlockNb 
     * @return bool isActive 
     */
    CI_DEVICE inline bool isActive(int itemNb, int itemPerBlockNb) {
        return (threadItemPos(itemPerBlockNb) < itemNb) && (TID < itemPerBlockNb);
    }

    /**
     * @brief Test if a thread have items to handle
     * 
     * @param itemNb 
     * @return bool isActive 
     */
    CI_DEVICE inline bool isActive(int itemNb) {
        return isActive(itemNb, itemPerBlockNb(itemNb));
    }
}

#endif //DOL_FUNCTION_H