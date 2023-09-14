#pragma once

#include <chrono>
#include <iostream>

#include "tools.hpp"

__device__ bool maskBlock(uint inVal, volatile uint * s_out, uint digit, uint inputSize,
                          const uint tid = threadIdx.x, const uint gid = blockDim.x * blockIdx.x + threadIdx.x)
{
    // clear mask
    s_out[tid] = 0;
    __syncthreads();

    // mask whole block
    bool mask = false;
    if (gid < inputSize) {
        mask = (inVal == digit);
        s_out[tid] = mask;
    }

    return mask;
}

__device__ void shuffleBlock(uint t_data, uint t_bits, uint * s_localPrefix, const uint * s_buckets, uint * s_data,
                             uint inputSize, uint * d_out, uint * d_localPrefixSums,
                             const uint tid = threadIdx.x, const uint gid = blockDim.x * blockIdx.x + threadIdx.x)
{
    if (gid<inputSize) {
        // calculate new positions
        uint t_prefix = s_localPrefix[tid];
        uint t_position = t_prefix + s_buckets[t_bits];
        __syncthreads();

        s_data[t_position] = t_data;
        s_localPrefix[t_position] = t_prefix; // local prefix is tied with element, move it also
        __syncthreads();

        // copy local shuffled array and local prefix sum back to global memory
        d_out[gid] = s_data[tid];
        d_localPrefixSums[gid] = s_localPrefix[tid];
    }
}

template<uint bits, uint buckets, uint blockSize>
__global__ void sortBlocks_kernel(const uint * d_in, uint * d_out, uint * d_blockSum, uint * d_localPrefixSums,
                                  size_t inputSize, uint index)
{
    __shared__ uint s_data[blockSize];
    __shared__ uint s_masked[blockSize];
    __shared__ uint s_buckets[blockSize];
    __shared__ uint s_localPrefix[blockSize];

    const uint tid = threadIdx.x;
    const uint gid = blockDim.x * blockIdx.x + threadIdx.x;

    // every block will process different part of input data
    // split input into blocks and copy each part into each block shared memory
    s_data[tid] = (gid < inputSize) ? d_in[gid] : 0;
    __syncthreads();

    uint t_data = s_data[tid];
    uint t_bits = getBits(t_data, index, bits);

    for (uint digit = 0; digit < buckets; ++digit) {
        bool t_mask = maskBlock(t_bits, s_masked, digit, inputSize);
        __syncthreads();

        // scan mask inplace to get Local Prefix for current digit
        blockScan<blockSize>(s_masked, &s_buckets[digit], &d_blockSum[digit * gridDim.x + blockIdx.x]);
        __syncthreads();

        // copy only relevant positions to create Local Prefix for all digits
        if (gid<inputSize)
            if (t_mask)
                s_localPrefix[tid] = s_masked[tid];
        __syncthreads();
    }

    blockScan<blockSize>(s_buckets);
    __syncthreads();

    // sort local block
    shuffleBlock(t_data, t_bits, s_localPrefix, s_buckets, s_data, inputSize, d_out, d_localPrefixSums);
}

template<uint bits, uint buckets>
__global__ void finalShuffle_kernel(const uint * d_in, uint * d_out, const uint * d_blockSum,
                                    const uint * d_localPrefixSums, size_t inputSize, uint index) {
    // Final position == P_d[n] + m
    // d = digit
    // n = input chunk --> blockIdx.x
    // m = local prefix sum
    // P - blockSum

    uint gid = blockDim.x * blockIdx.x + threadIdx.x;

    if (gid<inputSize) {
        uint t_data = d_in[gid];                        // d = digit
        uint t_prefix = d_localPrefixSums[gid];         // m = local prefix sum
        uint t_mask = getBits(t_data, index, bits);
        uint t_newPos = d_blockSum[gridDim.x * t_mask + blockIdx.x] + t_prefix;     // P_d[n] + m
        __syncthreads();

        d_out[t_newPos] = t_data;
    }
}


template<uint bitCount, uint blockSize>
void radixSortGPU(uint * d_in, size_t inputSize)
{
    constexpr uint typeSize = sizeof(uint) * 8;
    constexpr uint buckets = 1 << bitCount;
    constexpr uint bitMask = buckets - 1;
    static_assert(bitCount > 0, "Bit count must be at least one");
    static_assert(bitCount <= typeSize, "Too many bits for that type size");
    static_assert(typeSize % bitCount == 0, "Type size must divisible by bit count");
    static_assert(buckets <= blockSize, "Can not have more buckets than block size yet");

    uint gridSize = inputSize / blockSize + (inputSize % blockSize != 0);
    uint blockSumSize = buckets * gridSize;   // for each block one set of buckets

    uint *d_blockSum;
    checkCudaError(cudaMalloc(&d_blockSum, sizeof(uint) * blockSumSize));
    uint *d_localPrefixSums;
    checkCudaError(cudaMalloc(&d_localPrefixSums, sizeof(uint) * inputSize));
    uint *d_tempOut;
    checkCudaError(cudaMalloc(&d_tempOut, sizeof(uint) * inputSize));

    for (uint index = 0; index<typeSize; index += bitCount) {
        if (isSorted<blockSize>(d_in, inputSize))
            break;

        sortBlocks_kernel<bitMask, buckets, blockSize><<<gridSize, blockSize>>>(d_in, d_tempOut,
                                                                                d_blockSum, d_localPrefixSums, inputSize, index);
        thrust::exclusive_scan(thrust::device, d_blockSum, d_blockSum+blockSumSize, d_blockSum);
        // input is locally shuffled d_tempOut, output will be in d_in
        // thanks to local shuffle the final writes won't be so random
        finalShuffle_kernel<bitMask, buckets><<<gridSize, blockSize>>>(d_tempOut, d_in, d_blockSum,
                                                                       d_localPrefixSums, inputSize, index);
    }
    checkCudaError(cudaFree(d_blockSum));
    checkCudaError(cudaFree(d_localPrefixSums));
    checkCudaError(cudaFree(d_tempOut));
}