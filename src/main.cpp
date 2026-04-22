#include <iostream>
#include "../include/matrix.h"
#include "../include/operations.h"

int main() {
    std::cout << "=== Fast Matrix Operations Demo ===\n\n";

    Matrix A(3, 3);
    A.data = {{1,2,3},{4,5,6},{7,8,9}};

    Matrix B(3, 3);
    B.data = {{9,8,7},{6,5,4},{3,2,1}};

    std::cout << "A + B:\n"; (A + B).print();
    std::cout << "A * B:\n"; (A * B).print();
    std::cout << "A transposed:\n"; A.transpose().print();
    std::cout << "Frobenius norm of A: " << A.frobenius_norm() << "\n";

    Matrix C = Matrix::random(3, 3, -1, 1);
    std::cout << "\nReLU:\n"; ops::relu(C).print();
    std::cout << "Sigmoid:\n"; ops::sigmoid(C).print();

    return 0;
}