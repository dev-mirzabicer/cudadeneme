#ifndef VECTOR_H
#define VECTOR_H

#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include "cuda_helpers.h"
#include "cuda_settings.h"

class Vector {
public:
    int size;
    float* data;
    float* d_data;

    Vector(int size);
    ~Vector();

    void initialize(float value);
    std::string to_string() const;
    void to_gpu() const;
    void from_gpu() const;

    Vector operator+(const Vector& other) const;
    Vector operator-(const Vector& other) const;
    Vector elementwise_multiply(const Vector& other) const;
    float dot(const Vector& other) const;

    static __global__ void add_vectors(const float* A, const float* B, float* C, int size);
    static __global__ void subtract_vectors(const float* A, const float* B, float* C, int size);
    static __global__ void elementwise_multiply_vectors(const float* A, const float* B, float* C, int size);
    static __global__ void dot_product(const float* A, const float* B, float* C, int size);
};

#endif // VECTOR_H
