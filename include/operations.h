#pragma once
#include "matrix.h"
#include <stdexcept>

namespace ops {

    // Element-wise operations
    Matrix hadamard(const Matrix& A, const Matrix& B) {
        if (A.rows != B.rows || A.cols != B.cols)
            throw std::invalid_argument("Dimensions mismatch");
        Matrix result(A.rows, A.cols);
        for (int i = 0; i < A.rows; i++)
            for (int j = 0; j < A.cols; j++)
                result.data[i][j] = A.data[i][j] * B.data[i][j];
        return result;
    }

    // ReLU activation
    Matrix relu(const Matrix& A) {
        Matrix result(A.rows, A.cols);
        for (int i = 0; i < A.rows; i++)
            for (int j = 0; j < A.cols; j++)
                result.data[i][j] = std::max(0.0, A.data[i][j]);
        return result;
    }

    // Sigmoid activation
    Matrix sigmoid(const Matrix& A) {
        Matrix result(A.rows, A.cols);
        for (int i = 0; i < A.rows; i++)
            for (int j = 0; j < A.cols; j++)
                result.data[i][j] = 1.0 / (1.0 + std::exp(-A.data[i][j]));
        return result;
    }

    // Matrix power (A^n)
    Matrix power(const Matrix& A, int n) {
        if (A.rows != A.cols)
            throw std::invalid_argument("Matrix must be square");
        Matrix result = Matrix::identity(A.rows);
        Matrix base = A;
        while (n > 0) {
            if (n % 2 == 1) result = result * base;
            base = base * base;
            n /= 2;
        }
        return result;
    }

    // Dot product of two vectors
    double dot(const Matrix& a, const Matrix& b) {
        if (a.cols != 1 || b.cols != 1 || a.rows != b.rows)
            throw std::invalid_argument("Must be column vectors of same size");
        double result = 0;
        for (int i = 0; i < a.rows; i++)
            result += a.data[i][0] * b.data[i][0];
        return result;
    }

    // Cache-friendly matrix multiplication (blocked)
    Matrix matmul_blocked(const Matrix& A, const Matrix& B, int block_size = 32) {
        if (A.cols != B.rows)
            throw std::invalid_argument("Dimensions mismatch");
        Matrix result(A.rows, B.cols);
        for (int i = 0; i < A.rows; i += block_size)
            for (int k = 0; k < A.cols; k += block_size)
                for (int j = 0; j < B.cols; j += block_size)
                    for (int ii = i; ii < std::min(i+block_size, A.rows); ii++)
                        for (int kk = k; kk < std::min(k+block_size, A.cols); kk++)
                            for (int jj = j; jj < std::min(j+block_size, B.cols); jj++)
                                result.data[ii][jj] += A.data[ii][kk] * B.data[kk][jj];
        return result;
    }
}