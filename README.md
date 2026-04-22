# Fast Matrix Operations (C++)

![Language](https://img.shields.io/badge/Language-C%2B%2B17-blue)
![Tests](https://img.shields.io/badge/Tests-20%20passing-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

High-performance matrix operations library in C++ with ML-focused activations,
blocked matrix multiplication, and comprehensive benchmarking.

## Operations Implemented

| Operation | Description |
|-----------|-------------|
| Addition / Subtraction | Element-wise |
| Matrix Multiplication | Naive + Cache-friendly blocked |
| Transpose | O(n^2) |
| Hadamard Product | Element-wise multiply |
| ReLU / Sigmoid | ML activation functions |
| Frobenius Norm | Matrix magnitude |
| Matrix Power | Fast exponentiation |
| Dot Product | Vector dot product |

## Benchmark Results

    Matrix 64x64:
      Naive multiply:   126 us
      Blocked multiply: 103 us
      Speedup: 1.22x

    Matrix 128x128:
      Naive multiply:   811 us
      Blocked multiply: 800 us

    Operations on 128x128:
      Transpose:  59 us
      ReLU:       43 us
      Sigmoid:    819 us
      Frobenius:  8 us

## Quick Start

    git clone https://github.com/0mohamed123/fast-matrix-ops.git
    cd fast-matrix-ops

    # Run demo
    g++ -std=c++17 -O2 -I include -o main src/main.cpp
    ./main

    # Run benchmark
    g++ -std=c++20 -O2 -I include -o benchmark src/benchmark.cpp
    ./benchmark

    # Run tests
    g++ -std=c++17 -O2 -I include -o test_runner tests/test_matrix.cpp
    ./test_runner

## Test Results

    20 passed | 0 failed

    Tests cover: addition, subtraction, scalar multiply,
    matrix multiply, dimension validation, transpose,
    identity matrix, norms, ReLU, sigmoid, hadamard

## Key Design Decisions

- Header-only library using C++ templates
- Cache-friendly blocked matrix multiplication
- Fast matrix power using binary exponentiation
- RAII with std::vector (no raw pointer memory leaks)
- ML-focused: ReLU, Sigmoid, Hadamard for neural networks

## Technologies

- C++17 / C++20
- STL (vector, random, chrono)
- No external dependencies