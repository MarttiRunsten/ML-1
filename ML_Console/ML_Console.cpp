#include <iostream>

#include "network.h"

int main()
{
    // Testing
    std::cout << "Start test run\n";
    Network* net = new Network;
    std::cout << "End test run\n";

    net->feedInput();
    

    /*std::cout << "Append 1 to top of A:\n";
    A.appendOne().print();

     double (*op_ReLU)(double) { &ReLu };
    std::cout << "Activation (elementwise ReLU):\n";
    A.eWise(op_ReLU).print(); */
    
}

