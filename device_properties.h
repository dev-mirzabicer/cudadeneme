#ifndef DEVICE_PROPERTIES_H
#define DEVICE_PROPERTIES_H

#include <cuda_runtime.h>
#include <iostream>

struct DeviceProperties {
    int maxThreadsPerBlock;
    int maxThreadsPerSM;
    int warpSize;

    DeviceProperties() {
        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, 0);
        if (err != cudaSuccess) {
            std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
        maxThreadsPerBlock = prop.maxThreadsPerBlock;
        maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
        warpSize = prop.warpSize;
    }
};

#endif // DEVICE_PROPERTIES_H
