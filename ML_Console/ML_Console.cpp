#include <iostream>

#include "network.h"

int main()
{
    // Testing

    Matrix A(4, 1);

    std::cout << "A:\n";
    A.print();

    std::cout << "Clearing A:\n";
    A.clear();
    A.print();

    /*std::cout << "Append 1 to top of A:\n";
    A.appendOne().print();

     double (*op_ReLU)(double) { &ReLu };
    std::cout << "Activation (elementwise ReLU):\n";
    A.eWise(op_ReLU).print(); */
    
}

