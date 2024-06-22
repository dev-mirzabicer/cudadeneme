#include "vector.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>

Vector::Vector(int size) : size(size) {
    size_t bytes = size * sizeof(float);
    cudaMallocHost((void**)&data, bytes);
    if (!data) {
        std::cerr << "Failed to allocate pinned host memory" << std::endl;
        exit(EXIT_FAILURE);
    }
    checkCuda(cudaMalloc(&d_data, bytes));
}

Vector::~Vector() {
    cudaFreeHost(data);
    checkCuda(cudaFree(d_data));
}

void Vector::initialize(float value) {
    for (int i = 0; i < size; ++i) {
        data[i] = value;
    }
    checkCuda(cudaMemcpy(d_data, data, size * sizeof(float), cudaMemcpyHostToDevice));
}

std::string Vector::to_string() const {
    std::string result;
    for (int i = 0; i < size; ++i) {
        result += std::to_string(data[i]) + " ";
    }
    result += "\n";
    return result;
}

void Vector::to_gpu() const {
    checkCuda(cudaMemcpy(d_data, data, size * sizeof(float), cudaMemcpyHostToDevice));
}

void Vector::from_gpu() const {
    checkCuda(cudaMemcpy(data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost));
}

Vector Vector::operator+(const Vector& other) const {
    if (size != other.size) {
        std::cerr << "Vector dimensions must match for addition" << std::endl;
        exit(EXIT_FAILURE);
    }

    Vector result(size);
    int numBlocks = (size + CudaSettings::blockSize - 1) / CudaSettings::blockSize;

    add_vectors<<<numBlocks, CudaSettings::blockSize>>>(d_data, other.d_data, result.d_data, size);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    result.from_gpu();
    return result;
}

Vector Vector::operator-(const Vector& other) const {
    if (size != other.size) {
        std::cerr << "Vector dimensions must match for subtraction" << std::endl;
        exit(EXIT_FAILURE);
    }

    Vector result(size);
    int numBlocks = (size + CudaSettings::blockSize - 1) / CudaSettings::blockSize;

    subtract_vectors<<<numBlocks, CudaSettings::blockSize>>>(d_data, other.d_data, result.d_data, size);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    result.from_gpu();
    return result;
}

Vector Vector::elementwise_multiply(const Vector& other) const {
    if (size != other.size) {
        std::cerr << "Vector dimensions must match for element-wise multiplication" << std::endl;
        exit(EXIT_FAILURE);
    }

    Vector result(size);
    int numBlocks = (size + CudaSettings::blockSize - 1) / CudaSettings::blockSize;

    elementwise_multiply_vectors<<<numBlocks, CudaSettings::blockSize>>>(d_data, other.d_data, result.d_data, size);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    result.from_gpu();
    return result;
}

float Vector::dot(const Vector& other) const {
    if (size != other.size) {
        std::cerr << "Vector dimensions must match for dot product" << std::endl;
        exit(EXIT_FAILURE);
    }

    float result;
    float* d_result;
    checkCuda(cudaMalloc(&d_result, sizeof(float)));

    int numBlocks = (size + CudaSettings::blockSize - 1) / CudaSettings::blockSize;
    dot_product<<<numBlocks, CudaSettings::blockSize>>>(d_data, other.d_data, d_result, size);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    checkCuda(cudaFree(d_result));

    return result;
}

__global__ void Vector::add_vectors(const float* A, const float* B, float* C, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        C[index] = A[index] + B[index];
    }
}

__global__ void Vector::subtract_vectors(const float* A, const float* B, float* C, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        C[index] = A[index] - B[index];
    }
}

__global__ void Vector::elementwise_multiply_vectors(const float* A, const float* B, float* C, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        C[index] = A[index] * B[index];
    }
}

__global__ void Vector::dot_product(const float* A, const float* B, float* C, int size) {
    __shared__ float cache[CudaSettings::blockSize];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;
    if (index < size) {
        temp = A[index] * B[index];
    }

    cache[cacheIndex] = temp;

    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0) {
        atomicAdd(C, cache[0]);
    }
}
