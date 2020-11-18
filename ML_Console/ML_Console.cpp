#include <iostream>

#include "matrix.h"

double ReLu(double a) {
    if (a < 0) {
        return 0;
    }
    return a;
}

int main()
{
    Matrix A(2, 4);

    std::cout << "A:\n";
    A.print();

    double (*op_ReLU)(double) { &ReLu };
    std::cout << "Activation (elementwise ReLU):\n";
    A.eWise(op_ReLU).print();
    
}

