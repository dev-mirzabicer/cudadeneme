#include "matrix.h"
#include <iostream>
#include "cuda_helpers.h"

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols) {
    size_t size = rows * cols * sizeof(float);
    cudaMallocHost((void**)&data, size);
    if (!data) {
        std::cerr << "Failed to allocate pinned host memory" << std::endl;
        exit(EXIT_FAILURE);
    }
    checkCuda(cudaMalloc(&d_data, size));
}

Matrix::~Matrix() {
    cudaFreeHost(data);
    checkCuda(cudaFree(d_data));
}


void Matrix::initialize(float value) {
    for (int i = 0; i < rows * cols; ++i) {
        data[i] = value;
    }
    checkCuda(cudaMemcpy(d_data, data, rows * cols * sizeof(float), cudaMemcpyHostToDevice));
}

std::string Matrix::to_string() const {
    std::string result;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result += std::to_string(data[i * cols + j]) + " ";
        }
        result += "\n";
    }
    return result;
}

void Matrix::to_gpu() const {
    checkCuda(cudaMemcpy(d_data, data, rows * cols * sizeof(float), cudaMemcpyHostToDevice));
}

void Matrix::from_gpu() const {
    checkCuda(cudaMemcpy(data, d_data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
}

Matrix Matrix::operator+(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        std::cerr << "Matrix dimensions must match for addition" << std::endl;
        exit(EXIT_FAILURE);
    }

    Matrix result(rows, cols);
    int size = rows * cols;
    int numBlocks = (size + CudaSettings::blockSize - 1) / CudaSettings::blockSize;

    add_matrices<<<numBlocks, CudaSettings::blockSize>>>(d_data, other.d_data, result.d_data, rows, cols);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    result.from_gpu();
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols != other.rows) {
        std::cerr << "Matrix dimensions must match for multiplication" << std::endl;
        exit(EXIT_FAILURE);
    }

    Matrix result(rows, other.cols);
    dim3 blockSize(CudaSettings::tileSize, CudaSettings::tileSize);
    dim3 numBlocks((other.cols + blockSize.x - 1) / blockSize.x,
                   (rows + blockSize.y - 1) / blockSize.y);

    multiply_matrices<<<numBlocks, blockSize>>>(d_data, other.d_data, result.d_data, rows, cols, other.cols);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    result.from_gpu();
    return result;
}

Vector Matrix::operator*(const Vector& vec) const {
    if (cols != vec.size) {
        std::cerr << "Matrix columns must match vector size for multiplication" << std::endl;
        exit(EXIT_FAILURE);
    }

    Vector result(rows);
    int numBlocks = (rows + CudaSettings::blockSize - 1) / CudaSettings::blockSize;

    multiply_matrix_vector<<<numBlocks, CudaSettings::blockSize>>>(d_data, vec.d_data, result.d_data, rows, cols);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    result.from_gpu();
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        std::cerr << "Matrix dimensions must match for subtraction" << std::endl;
        exit(EXIT_FAILURE);
    }

    Matrix result(rows, cols);
    int size = rows * cols;
    int numBlocks = (size + CudaSettings::blockSize - 1) / CudaSettings::blockSize;

    subtract_matrices<<<numBlocks, CudaSettings::blockSize>>>(d_data, other.d_data, result.d_data, rows, cols);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    result.from_gpu();
    return result;
}

__global__ void Matrix::add_matrices(const float* A, const float* B, float* C, int rows, int cols) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < rows * cols) {
        C[index] = A[index] + B[index];
    }
}

__global__ void Matrix::multiply_matrices(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB) {
    __shared__ float tileA[CudaSettings::tileSize][CudaSettings::tileSize];
    __shared__ float tileB[CudaSettings::tileSize][CudaSettings::tileSize];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float value = 0;

    for (int t = 0; t < (colsA + CudaSettings::tileSize - 1) / CudaSettings::tileSize; ++t) {
        if (row < rowsA && t * CudaSettings::tileSize + threadIdx.x < colsA)
            tileA[threadIdx.y][threadIdx.x] = A[row * colsA + t * CudaSettings::tileSize + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0;

        if (col < colsB && t * CudaSettings::tileSize + threadIdx.y < colsA)
            tileB[threadIdx.y][threadIdx.x] = B[(t * CudaSettings::tileSize + threadIdx.y) * colsB + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        for (int k = 0; k < CudaSettings::tileSize; ++k)
            value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < rowsA && col < colsB)
        C[row * colsB + col] = value;
}

__global__ void Matrix::subtract_matrices(const float* A, const float* B, float* C, int rows, int cols) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < rows * cols) {
        C[index] = A[index] - B[index];
    }
}

__global__ void Matrix::multiply_matrix_vector(const float* A, const float* B, float* C, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float value = 0;
        for (int col = 0; col < cols; ++col) {
            value += A[row * cols + col] * B[col];
        }
        C[row] = value;
    }
}

__global__ void Matrix::transpose_matrix(const float* A, float* B, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        B[x * rows + y] = A[y * cols + x];
    }
}

Matrix Matrix::transpose() const {
    Matrix result(cols, rows); // Rows and cols are swapped

    dim3 blockSize(CudaSettings::tileSize, CudaSettings::tileSize);
    dim3 numBlocks((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

    transpose_matrix<<<numBlocks, blockSize>>>(d_data, result.d_data, rows, cols);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    result.from_gpu();
    return result;
}

__global__ void Matrix::elementwise_multiply(const float* A, const float* B, float* C, int rows, int cols) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < rows * cols) {
        C[index] = A[index] * B[index];
    }
}

Matrix Matrix::elementwise_multiply(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        std::cerr << "Matrix dimensions must match for element-wise multiplication" << std::endl;
        exit(EXIT_FAILURE);
    }

    Matrix result(rows, cols);
    int size = rows * cols;
    int numBlocks = (size + CudaSettings::blockSize - 1) / CudaSettings::blockSize;

    elementwise_multiply<<<numBlocks, CudaSettings::blockSize>>>(d_data, other.d_data, result.d_data, rows, cols);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    result.from_gpu();
    return result;
}
