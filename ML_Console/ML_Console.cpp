#include <iostream>

#include "network.h"

int main()
{
    // Testing

    Matrix A(4, 1);

    std::vector<std::vector<double>> c;
    for (int i = 0; i < 4; i++) {
        std::vector<double> r;
        for (int j = 0; j < 2; j++) {
            r.push_back((double)i*j);
        }
        c.push_back(r);
    }
    Matrix B(c);

    std::cout << "A:\n";
    A.print();

    std::cout << "B:\n";
    B.print();

    std::cout << "B is of size " << B.size().first << "x" << B.size().second << ".\n";

    /*std::cout << "Append 1 to top of A:\n";
    A.appendOne().print();

     double (*op_ReLU)(double) { &ReLu };
    std::cout << "Activation (elementwise ReLU):\n";
    A.eWise(op_ReLU).print(); */
    
}

