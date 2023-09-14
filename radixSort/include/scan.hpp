#pragma once

#include "tools.hpp"

template <uint blockSize>
__device__ void blockScan(uint *s_data, uint * sum1 = nullptr, uint * sum2 = nullptr, uint tid = threadIdx.x);

#include "scan.tpp"