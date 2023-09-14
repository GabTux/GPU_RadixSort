#pragma once

#include <chrono>
#include <iostream>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include "tools.hpp"

enum sortType {
    STD,
    Thrust,
    Func
};

template<sortType type, typename Function>
void measure_time_cpu(Function func, const char* name, uint * inData, size_t size)
{
    auto start = std::chrono::steady_clock::now();
    if constexpr (type == sortType::STD)
        std::sort(inData, inData + size);
    else
        func(inData, size);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << std::fixed << name << ": " << elapsed.count() << "s" << std::endl;
}

template<sortType type, typename Function>
void measure_time_gpu(Function func, const char* name, uint * inData, size_t size) {
    uint *d_inGpu;
    checkCudaError(cudaMalloc(&d_inGpu, sizeof(uint) * size));
    checkCudaError(cudaMemcpy(d_inGpu, inData, sizeof(uint) * size, cudaMemcpyHostToDevice));
    auto start = std::chrono::steady_clock::now();
    if constexpr (type == sortType::Thrust)
        thrust::sort(thrust::device, d_inGpu, d_inGpu + size);
    else
        func(d_inGpu, size);
    checkCudaError(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << std::fixed << name << ": " << elapsed.count() << "s" << std::endl;

    checkCudaError(cudaMemcpy(inData, d_inGpu, sizeof(uint) * size, cudaMemcpyDeviceToHost));
    checkCudaError(cudaFree(d_inGpu));
}