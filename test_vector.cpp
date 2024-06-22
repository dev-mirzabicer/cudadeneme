#include "vector.h"
#include <iostream>

void test_vector_addition() {
    Vector A(5);
    A.initialize(1.0f);
    Vector B(5);
    B.initialize(2.0f);

    Vector C = A + B;
    std::cout << "Vector A:\n" << A.to_string() << std::endl;
    std::cout << "Vector B:\n" << B.to_string() << std::endl;
    std::cout << "Vector C (A + B):\n" << C.to_string() << std::endl;
}

void test_vector_subtraction() {
    Vector A(5);
    A.initialize(2.0f);
    Vector B(5);
    B.initialize(1.0f);

    Vector C = A - B;
    std::cout << "Vector A:\n" << A.to_string() << std::endl;
    std::cout << "Vector B:\n" << B.to_string() << std::endl;
    std::cout << "Vector C (A - B):\n" << C.to_string() << std::endl;
}

void test_elementwise_multiplication() {
    Vector A(5);
    A.initialize(2.0f);
    Vector B(5);
    B.initialize(3.0f);

    Vector C = A.elementwise_multiply(B);
    std::cout << "Vector A:\n" << A.to_string() << std::endl;
    std::cout << "Vector B:\n" << B.to_string() << std::endl;
    std::cout << "Vector C (A .* B):\n" << C.to_string() << std::endl;
}

void test_dot_product() {
    Vector A(5);
    A.initialize(2.0f);
    Vector B(5);
    B.initialize(3.0f);

    float dotProduct = A.dot(B);
    std::cout << "Vector A:\n" << A.to_string() << std::endl;
    std::cout << "Vector B:\n" << B.to_string() << std::endl;
    std::cout << "Dot Product (A . B):\n" << dotProduct << std::endl;
}

int main() {
    test_vector_addition();
    test_vector_subtraction();
    test_elementwise_multiplication();
    test_dot_product();

    return 0;
}
