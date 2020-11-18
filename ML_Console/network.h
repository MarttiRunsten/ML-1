#pragma once

#include "matrix.h"
#include "activations.h"

class Network;

class Layer
{
public:
	Layer(Network* parent, int size, double(*activation)(double));
	~Layer();

	void feedForward(Matrix In);

	Matrix calculateD(Matrix& D_upper);

private:
	Network* net_;
	Layer* next_;
	Layer* prev_;

	Matrix W_;
	Matrix D_;

	void updateW();
};

class Network
{
public:
	Network();
	~Network();

	void feedInput();
	void backpropagate(Matrix& O);

private:
	double lambda_;
	double rho_;


};

