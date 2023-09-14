#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <bitset>
#include <random>
#include "pcg_random.hpp"

typedef unsigned int uint;

// Inspired from here: https://stackoverflow.com/a/14038590
#define checkCudaError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

void printArray(const char *text, uint *array, uint size, bool binary = false)
{
    std::cout << text << ": ";
    for (size_t i = 0; i < size; ++i) {
        if (binary)
            std::cout << std::bitset<sizeof(array[i])*8>(array[i]) << " ";
        else
            std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}

uint *generateInput(size_t size, bool random, uint max) {
    uint * result = new uint[size];
    pcg64 rng;
    if (random)
        rng.seed(pcg_extras::seed_seq_from<std::random_device>());
    else
        rng.seed(PCG_128BIT_CONSTANT(0xC7C8709C9626D159ULL, 0x675BB824D76E9146ULL));
    for (size_t i = 0; i < size; ++i) {
        result[i] = rng(max);
    }
    return result;
}

bool checkResults(uint *refArr, uint *resArr, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        if (refArr[i] != resArr[i]) {
            std::cout << "MISMATCH DETECTED, position: " << i << std::endl;
            std::cout << refArr[i] << " vs " << resArr[i] << std::endl << std::endl;
            #ifndef NDEBUG
            printArray("refArray", refArr, size);
            std::cout << "------------" << std::endl;
            printArray("resArray", resArr, size);
            #endif
            return false;
        }
    }
    return true;
}

#ifndef NDEBUG
__device__ void printArrayDev(const char *text, uint *array, size_t size, uint id = 0)
{
    __syncthreads();
    uint gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid == id) {
        printf("%s: ", text);
        for (size_t i = 0; i < size; ++i) {
            printf("%u ", array[i]);
        }
        printf("\n");
    }
    __syncthreads();
}
#endif
