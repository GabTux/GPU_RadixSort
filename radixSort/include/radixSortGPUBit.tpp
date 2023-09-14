#pragma once

__global__ void extractBits_kernel(uint* d_data, uint* d_bits, uint bit, size_t size)
{
    uint gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid < size) {
        uint t_bit = getBits(d_data[gid], bit, 0x01);
        d_bits[gid] = t_bit;
    }
}


__global__ void shuffle_kernel(const uint* d_data, uint* d_out, const uint * prefixSum, const uint* bits, size_t size)
{
    uint gid = blockDim.x * blockIdx.x + threadIdx.x;

    if (gid < size) {
        uint t_prefix = prefixSum[gid];
        uint t_zerosCount = size - prefixSum[size - 1];
        uint t_value = d_data[gid];

        if (bits[gid]) {
            d_out[t_prefix - 1 + t_zerosCount] = t_value;
        } else {
            d_out[gid - t_prefix] = t_value;
        }
    }
}

template <uint blockSize>
void radixSortGPUBit(uint *d_data, size_t size) {
    // prepare DEV memory
    uint* d_bits;
    checkCudaError(cudaMalloc(&d_bits, size * sizeof(uint)));
    uint* d_prefixSum;
    checkCudaError(cudaMalloc(&d_prefixSum, size * sizeof(uint)));
    uint* d_tempOut;
    checkCudaError(cudaMalloc(&d_tempOut, size * sizeof(uint)));

    uint gridSize = size / blockSize + (size % blockSize != 0);
    constexpr uint typeSize = sizeof(uint) * 8;

    for (uint bit = 0; bit < typeSize; ++bit)
    {
        if (isSorted<blockSize>(d_data, size))
            break;
        extractBits_kernel<<<gridSize, blockSize>>>(d_data, d_bits, bit, size);
        thrust::inclusive_scan(thrust::device, d_bits, d_bits+size, d_prefixSum);
        shuffle_kernel<<<gridSize, blockSize>>>(d_data, d_tempOut, d_prefixSum, d_bits, size);
        checkCudaError(cudaMemcpy(d_data, d_tempOut, size * sizeof(uint), cudaMemcpyDeviceToDevice));
    }
    checkCudaError(cudaFree(d_bits));
    checkCudaError(cudaFree(d_prefixSum));
    checkCudaError(cudaFree(d_tempOut));
}
