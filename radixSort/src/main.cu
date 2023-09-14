#include <iostream>
#include <chrono>
#include <algorithm>
#include <string>

#include "radixSort.hpp"
#include "tools.hpp"
#include "measureTime.hpp"


int main(int argc, char * argv[]) {
    if (argc != 2) {
        std::cout << "Invalid arguments" << std::endl << "Use: " << argv[0] << " <num of elements>" << std::endl;
        return EXIT_FAILURE;
    }
    size_t size = std::stoull(argv[1]);
    uint * inputArr = generateInput(size, false, std::numeric_limits<uint>::max());
    //uint * inputArr = generateInput(size, false, 1000);
    uint * testArr = new uint[size];
    uint * sortArr = new uint[size];
    memcpy(sortArr, inputArr, size * sizeof(uint));

    // std::sort sort
    measure_time_cpu<sortType::STD>(std::function<void()>(), "std::sort", sortArr, size);

    // Thrust sort
    memcpy(testArr, inputArr, size * sizeof(uint));
    measure_time_gpu<sortType::Thrust>(std::function<void()>(), "thrust::sort", sortArr, size);

    // CPU radix sort 8 bits
    memcpy(testArr, inputArr, size * sizeof(uint));
    measure_time_cpu<sortType::Func>(radixSortCPU<8>, "radixSortCPU 8 bits", testArr, size);
    if (!checkResults(sortArr, testArr, size)) {
        throw std::runtime_error("Mismatch");
    }

    // GPU radix sort 2 bits
    memcpy(testArr, inputArr, size * sizeof(uint));
    measure_time_gpu<sortType::Func>(radixSortGPU<2, 256>, "radixSortGPU 2 bits 256", testArr, size);
    if (!checkResults(sortArr, testArr, size)) {
        throw std::runtime_error("Mismatch");
    }

    delete [] sortArr;
    delete [] testArr;
    delete [] inputArr;
}