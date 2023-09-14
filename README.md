# GPU_RadixSort
I've created a Radix sort algorithm specifically designed for GPUs using CUDA. The final product's performance has been evaluated by comparing it to traditional CPU-based Radix sort, std::sort, and thrust::sort.

### Usage
1) Build:
```
make
```
2) Run:
```
./testRadix <num of elements>
```
Example of running for 256 million random elements <br>
CPU: AMD Ryzen 9 4900HS <br>
GPU: NVidia GeForce GTX 1660 Ti <br>
```
❯ ./testRadix 256000000
std::sort: 17.942140s
thrust::sort: 0.159063s
radixSortCPU 8 bits: 3.010192s
radixSortGPU 2 bits 256: 1.714856s
```
The results show that the GPU-based Radix sort is 10 times faster on this hardware than std::sort.
