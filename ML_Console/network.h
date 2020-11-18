#pragma once

#include "matrix.h"

class Network;

class Layer
{
public:
	Layer(Network* parent, int input_size, int layer_size, Base* activation);
	~Layer();

	void feedForward(Matrix& In);

	void backpropagate(Matrix& D_upper, Matrix& W_upper);

	std::pair<int, int> size();

private:
	Network* net_;
	Layer* next_;
	Layer* prev_;
	Base* activ_;
	int isize_;
	int lsize_;

	Matrix W_;
	Matrix A_;
	Matrix O_;
	Matrix D_;

	void updateW();
};

class Network
{
public:
	Network();
	~Network();

	void feedInput();
	void receiveOutput(Matrix& O);

	void backpropagate();
	void bpDone();

private:
	double lambda_;
	double rho_;
};

