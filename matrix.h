#ifndef MATRIX_H
#define MATRIX_H

#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include "cuda_settings.h"
#include "vector.h"

class Matrix {
public:
    int rows;
    int cols;
    float* data;
    float* d_data;

    Matrix(int rows, int cols);
    ~Matrix();

    void initialize(float value);
    std::string to_string() const;
    void to_gpu() const;
    void from_gpu() const;

    Matrix operator+(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Vector operator*(const Vector& vec) const;
    
    static __global__ void subtract_matrices(const float* A, const float* B, float* C, int rows, int cols);
    static __global__ void add_matrices(const float* A, const float* B, float* C, int rows, int cols);
    static __global__ void multiply_matrices(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB);
    static __global__ void multiply_matrix_vector(const float* A, const float* B, float* C, int rows, int cols);

    Matrix transpose() const;
    static __global__ void transpose_matrix(const float* A, float* B, int rows, int cols);

    Matrix elementwise_multiply(const Matrix& other) const;
    static __global__ void elementwise_multiply(const float* A, const float* B, float* C, int rows, int cols);

};

#endif // MATRIX_H
