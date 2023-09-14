#pragma once

#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <algorithm>

#include "tools.hpp"
#include "scan.hpp"

/* RADIX SORT CPU */
template <uint buckets>
void radixSortCPU(uint* array, size_t size);
#include "radixSortCPU.tpp"

/* DEVICE COMMON FUNCTIONS */
__device__ uint inline getBits(const uint &input, uint &index, const uint &bits) {
    return (input >> index) & bits;
}

template <uint blockSize>
bool isSorted(uint * d_in, size_t inputSize) {
    size_t blockSizeCheck = (inputSize > blockSize) ? blockSize : inputSize;
    bool blockSorted = thrust::is_sorted(thrust::device, d_in, d_in+blockSizeCheck);
    if (blockSorted) {
        return thrust::is_sorted(thrust::device, d_in, d_in+inputSize);
    } else {
        return false;
    }
}

/* RADIX SORT GPU BIT */
template <uint blockSize>
void radixSortGPUBit(uint *d_data, size_t size);
#include "radixSortGPUBit.tpp"

/* RADIX SORT GPU */
template <uint bitCount, uint blockSize>
void radixSortGPU(uint *d_in, size_t inputSize);
#include "radixSortGPU.tpp"