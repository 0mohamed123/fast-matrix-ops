#include <iostream>
#include <cassert>
#include <cmath>
#include <string>
#include "../include/matrix.h"
#include "../include/operations.h"

int passed = 0, failed = 0;

void test(bool condition, const std::string& name) {
    if (condition) { std::cout << "  [PASS] " << name << "\n"; passed++; }
    else { std::cout << "  [FAIL] " << name << "\n"; failed++; }
}

bool approx_equal(double a, double b, double tol = 1e-9) {
    return std::abs(a - b) < tol;
}

void test_basic() {
    std::cout << "\n[Basic Operations]\n";
    Matrix A(2, 2);
    A.data = {{1,2},{3,4}};
    Matrix B(2, 2);
    B.data = {{5,6},{7,8}};

    Matrix C = A + B;
    test(C.data[0][0] == 6 && C.data[1][1] == 12, "addition correct");

    Matrix D = A - B;
    test(D.data[0][0] == -4 && D.data[1][1] == -4, "subtraction correct");

    Matrix E = A * 2.0;
    test(E.data[0][0] == 2 && E.data[1][1] == 8, "scalar multiply correct");
}

void test_multiply() {
    std::cout << "\n[Matrix Multiplication]\n";
    Matrix A(2, 3);
    A.data = {{1,2,3},{4,5,6}};
    Matrix B(3, 2);
    B.data = {{7,8},{9,10},{11,12}};

    Matrix C = A * B;
    test(C.rows == 2 && C.cols == 2, "output shape correct");
    test(C.data[0][0] == 58 && C.data[0][1] == 64, "first row correct");
    test(C.data[1][0] == 139 && C.data[1][1] == 154, "second row correct");

    bool threw = false;
    try { Matrix X(2,3); Matrix Y(2,3); X * Y; } catch(...) { threw = true; }
    test(threw, "dimension mismatch throws");
}

void test_transpose() {
    std::cout << "\n[Transpose]\n";
    Matrix A(2, 3);
    A.data = {{1,2,3},{4,5,6}};
    Matrix T = A.transpose();
    test(T.rows == 3 && T.cols == 2, "shape correct");
    test(T.data[0][0] == 1 && T.data[2][1] == 6, "values correct");
    test(A.transpose().transpose() == A, "double transpose = original");
}

void test_identity() {
    std::cout << "\n[Identity]\n";
    Matrix I = Matrix::identity(3);
    Matrix A = Matrix::random(3, 3);
    test((A * I) == A, "A * I = A");
    test((I * A) == A, "I * A = A");
}

void test_norms() {
    std::cout << "\n[Norms & Stats]\n";
    Matrix A(2, 2);
    A.data = {{3,0},{4,0}};
    test(approx_equal(A.frobenius_norm(), 5.0), "frobenius norm correct");
    test(approx_equal(A.sum(), 7.0), "sum correct");
    test(approx_equal(A.mean(), 1.75), "mean correct");
}

void test_activations() {
    std::cout << "\n[Activations]\n";
    Matrix A(2, 2);
    A.data = {{-1, 2},{-3, 4}};

    Matrix R = ops::relu(A);
    test(R.data[0][0] == 0 && R.data[0][1] == 2, "ReLU negatives zeroed");
    test(R.data[1][0] == 0 && R.data[1][1] == 4, "ReLU positives kept");

    Matrix S = ops::sigmoid(A);
    test(S.data[0][0] > 0 && S.data[0][0] < 0.5, "sigmoid(-1) < 0.5");
    test(S.data[0][1] > 0.5 && S.data[0][1] < 1.0, "sigmoid(2) > 0.5");
}

void test_hadamard() {
    std::cout << "\n[Hadamard]\n";
    Matrix A(2, 2); A.data = {{1,2},{3,4}};
    Matrix B(2, 2); B.data = {{2,3},{4,5}};
    Matrix C = ops::hadamard(A, B);
    test(C.data[0][0] == 2 && C.data[1][1] == 20, "hadamard correct");
}

int main() {
    std::cout << "Running all tests...\n";
    test_basic();
    test_multiply();
    test_transpose();
    test_identity();
    test_norms();
    test_activations();
    test_hadamard();

    std::cout << "\n==============================\n";
    std::cout << "  " << passed << " passed | " << failed << " failed\n";
    std::cout << "==============================\n";
    return failed > 0 ? 1 : 0;
}