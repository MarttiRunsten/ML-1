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
    Matrix A(4, 1);

    std::cout << "A:\n";
    A.print();

    std::cout << "Append 1 to top of A:\n";
    A.appendOne().print();

    /* double (*op_ReLU)(double) { &ReLu };
    std::cout << "Activation (elementwise ReLU):\n";
    A.eWise(op_ReLU).print(); */
    
}

