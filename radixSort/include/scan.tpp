#pragma once

// reused from short job 1 :)

// volatile to prevent compiler optimizations
// use Hillis Steele algorithm to do warp scan
// use warp-level primitives for speeeed
// result in s_data will be inclusive scan
__device__ void warpScan(volatile uint *s_data,
                         uint tid = threadIdx.x)
{
    // thread id in warp
    uint twid = tid & 31;
    uint val = s_data[tid];

    unsigned mask = 0xffffffff;
    #pragma unroll
    for (uint offset = 1; offset < 32; offset <<= 1) {
        uint up_val = __shfl_up_sync(mask, val, offset);
        if (twid >= offset)
            val += up_val;
    }
    s_data[tid] = val;
}

template <uint blockSize>
__device__ void blockScan(uint *s_data, uint * sum1, uint * sum2, uint tid)
{
    uint twid = tid & 31;
    // warp id inside block
    uint wid = tid >> 5;

    __shared__ uint s_warpRes[32];

    // split input to warps and scan it inside warps
    warpScan(s_data, tid);
    __syncthreads();

    // get warp sums
    if (twid == 31)
        s_warpRes[wid] = s_data[tid];
    __syncthreads();

    // use first warp to scan the sums
    // maximum of warps is 32, so we can always scan it only by one warp
    if (wid == 0)
        warpScan(s_warpRes, tid);
    __syncthreads();

    // Add sums to final result
    if (wid > 0)
        s_data[tid] += s_warpRes[wid - 1];
    __syncthreads();

    // shift final result and write sums
    uint val = s_data[tid];
    __syncthreads();

    if (tid < blockSize - 1) {
        s_data[tid + 1] = val;
    } else {
        s_data[0] = 0;
        if ((sum1 != nullptr) && (sum2 != nullptr)) {
            *sum1 = val;
            *sum2 = val;
        }
    }
}