#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H

#include <iostream>
#include <cuda_runtime.h>

inline void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        exit(result);
    }
}

#endif // CUDA_HELPERS_H
