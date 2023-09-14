#pragma once

#include <cmath>
#include <bitset>
#include <cstring>
#include <numeric>
#include <algorithm>

uint inline extractBits(const uint & input, const uint & bits, uint index)
{
    return (input >> index) & bits;
}

template <uint bits, uint buckets>
void sortDigit(uint * array, uint * out, size_t size, uint index)
{
    uint prefixSum[buckets] = {0};

    // histogram
    for (size_t i = 0; i < size; ++i) {
        prefixSum[extractBits(array[i], bits, index)]++;
    }

    // Exclusive prefix sum
    std::exclusive_scan(prefixSum, prefixSum+buckets, prefixSum, 0);

    // shuffle_kernel elements
    for (size_t i = 0; i < size; ++i) {
        out[prefixSum[extractBits(array[i], bits, index)]++] = array[i];
    }
}

template <uint bitCount>
void radixSortCPU(uint *array, size_t size)
{
    constexpr uint typeSize = sizeof(uint) * 8;
    constexpr uint buckets = 1 << bitCount;
    constexpr uint bitMask = buckets - 1;
    static_assert(bitCount > 0, "Bit count must be at least one");
    static_assert(bitCount <= typeSize, "Too many bits for that type size");
    static_assert(typeSize % bitCount == 0, "Type size must divisible by bit count");

    uint * out = new uint[size];

    bool swapped = false;
    for (size_t index = 0; index < typeSize; index += bitCount) {
        sortDigit<bitMask, buckets>(array, out, size, index);
        std::swap(array, out);
        swapped = !swapped;
    }
    if (swapped)
        std::swap(array, out);

    delete[] out;
}