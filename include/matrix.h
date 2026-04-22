#pragma once
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <random>

class Matrix {
public:
    int rows, cols;
    std::vector<std::vector<double>> data;

    Matrix(int rows, int cols, double val = 0.0)
        : rows(rows), cols(cols), data(rows, std::vector<double>(cols, val)) {}

    static Matrix random(int rows, int cols, double min = 0.0, double max = 1.0) {
        Matrix m(rows, cols);
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(min, max);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m.data[i][j] = dist(rng);
        return m;
    }

    static Matrix identity(int n) {
        Matrix m(n, n);
        for (int i = 0; i < n; i++) m.data[i][i] = 1.0;
        return m;
    }

    double& at(int i, int j) { return data[i][j]; }
    const double& at(int i, int j) const { return data[i][j]; }

    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols)
            throw std::invalid_argument("Matrix dimensions mismatch");
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result.data[i][j] = data[i][j] + other.data[i][j];
        return result;
    }

    Matrix operator-(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols)
            throw std::invalid_argument("Matrix dimensions mismatch");
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result.data[i][j] = data[i][j] - other.data[i][j];
        return result;
    }

    Matrix operator*(double scalar) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result.data[i][j] = data[i][j] * scalar;
        return result;
    }

    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows)
            throw std::invalid_argument("Matrix dimensions mismatch for multiplication");
        Matrix result(rows, other.cols);
        for (int i = 0; i < rows; i++)
            for (int k = 0; k < cols; k++)
                for (int j = 0; j < other.cols; j++)
                    result.data[i][j] += data[i][k] * other.data[k][j];
        return result;
    }

    bool operator==(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) return false;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                if (std::abs(data[i][j] - other.data[i][j]) > 1e-9) return false;
        return true;
    }

    Matrix transpose() const {
        Matrix result(cols, rows);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result.data[j][i] = data[i][j];
        return result;
    }

    double sum() const {
        double s = 0;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                s += data[i][j];
        return s;
    }

    double mean() const { return sum() / (rows * cols); }

    double frobenius_norm() const {
        double s = 0;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                s += data[i][j] * data[i][j];
        return std::sqrt(s);
    }

    void print() const {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++)
                std::cout << data[i][j] << "\t";
            std::cout << "\n";
        }
    }
};