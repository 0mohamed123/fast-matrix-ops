#include <iostream>
#include <chrono>
#include "../include/matrix.h"
#include "../include/operations.h"

using namespace std::chrono;

template<typename F>
long long measure(F func) {
    auto start = high_resolution_clock::now();
    func();
    auto end = high_resolution_clock::now();
    return duration_cast<microseconds>(end - start).count();
}

int main() {
    std::cout << "============================================\n";
    std::cout << "   Fast Matrix Operations - Benchmark\n";
    std::cout << "============================================\n\n";

    for (int n : {64, 128, 256}) {
        Matrix A = Matrix::random(n, n);
        Matrix B = Matrix::random(n, n);

        auto t1 = measure([&]() { A * B; });
        auto t2 = measure([&]() { ops::matmul_blocked(A, B); });

        std::cout << "Matrix " << n << "x" << n << ":\n";
        std::cout << "  Naive multiply:   " << t1 << " us\n";
        std::cout << "  Blocked multiply: " << t2 << " us\n";
        std::cout << "  Speedup: " << (double)t1/t2 << "x\n\n";
    }

    // Operations benchmark
    int n = 128;
    Matrix A = Matrix::random(n, n, -1, 1);

    std::cout << "Operations on " << n << "x" << n << " matrix:\n";
    std::cout << "  Transpose:  " << measure([&]() { A.transpose(); }) << " us\n";
    std::cout << "  ReLU:       " << measure([&]() { ops::relu(A); }) << " us\n";
    std::cout << "  Sigmoid:    " << measure([&]() { ops::sigmoid(A); }) << " us\n";
    std::cout << "  Frobenius:  " << measure([&]() { A.frobenius_norm(); }) << " us\n";

    std::cout << "\n============================================\n";
    return 0;
}