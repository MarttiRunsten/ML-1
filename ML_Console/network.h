#pragma once

#include "matrix.h"
#include <cmath>

class Network;

class Layer
{
public:
	Layer(Network* parent);
	~Layer();

	void setup();
	void setNext(Layer* n);
	void setPrev(Layer* p);

	void feedForward(mPtr In);

	void backpropagate(mPtr D_upper, mPtr W_upper);

	std::pair<int, int> size();
	Layer* getPrev();
	mPtr getI();

private:
	Network* net_;
	Layer* next_;
	Layer* prev_;
	Base* activ_;
	int isize_;
	int lsize_;
	bool isBin_;
	double leak_;

	mPtr W_; // Bias|Weights (Matrix)
	mPtr A_; // Sum of weighted inputs and bias per neuron (Vector)
	mPtr O_; // Output (Vector)
	mPtr D_; // Deltas (Vector)
	mPtr I_; // Inputs (Vector)

	void updateW();
};

class Network
{
public:
	Network();
	~Network();

	void feedInput();
	void receiveOutput(mPtr O);
	void bpDone();

	double getL();
	double getR();

	Layer* getOut();

	Base* RELU;
	BinBase* LRELU;
	Base* SIGMOID;

private:
	double lambda_;
	double rho_;

	int iter_;
	int train_; // Not the best way. This is data-independent.

	Layer* in_;
	Layer* out_;

	double(*sin_)(double) { &sin };

	void setHypers();

	

	void test();
};

