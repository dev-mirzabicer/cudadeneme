#include "matrix.h"
#include <iostream>

void test_matrix_addition() {
    Matrix A(2, 2);
    A.initialize(1.0f);
    Matrix B(2, 2);
    B.initialize(2.0f);

    Matrix C = A + B;
    std::cout << "Matrix A:\n" << A.to_string() << std::endl;
    std::cout << "Matrix B:\n" << B.to_string() << std::endl;
    std::cout << "Matrix C (A + B):\n" << C.to_string() << std::endl;
}

void test_matrix_subtraction() {
    Matrix A(2, 2);
    A.initialize(3.0f);
    Matrix B(2, 2);
    B.initialize(2.0f);

    Matrix C = A - B;
    std::cout << "Matrix A:\n" << A.to_string() << std::endl;
    std::cout << "Matrix B:\n" << B.to_string() << std::endl;
    std::cout << "Matrix C (A - B):\n" << C.to_string() << std::endl;
}

void test_matrix_multiplication() {
    Matrix A(2, 3);
    A.initialize(1.0f);
    Matrix B(3, 2);
    B.initialize(2.0f);

    Matrix C = A * B;
    std::cout << "Matrix A:\n" << A.to_string() << std::endl;
    std::cout << "Matrix B:\n" << B.to_string() << std::endl;
    std::cout << "Matrix C (A * B):\n" << C.to_string() << std::endl;
}

void test_matrix_transposition() {
    Matrix A(2, 3);
    A.initialize(1.0f);

    Matrix B = A.transpose();
    std::cout << "Matrix A:\n" << A.to_string() << std::endl;
    std::cout << "Matrix B (A^T):\n" << B.to_string() << std::endl;
}

void test_elementwise_multiplication() {
    Matrix A(2, 2);
    A.initialize(2.0f);
    Matrix B(2, 2);
    B.initialize(3.0f);

    Matrix C = A.elementwise_multiply(B);
    std::cout << "Matrix A:\n" << A.to_string() << std::endl;
    std::cout << "Matrix B:\n" << B.to_string() << std::endl;
    std::cout << "Matrix C (A .* B):\n" << C.to_string() << std::endl;
}

int main() {
    test_matrix_addition();
    test_matrix_subtraction();
    test_matrix_multiplication();
    test_matrix_transposition();
    test_elementwise_multiplication();

    return 0;
}
