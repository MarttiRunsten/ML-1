#include "activations.h"

double Base::activation(double a) {
	return a;
}

double Base::differential(double a) {
	return 0;
}

double ReLU::activation(double a) {
	if (a < 0) {
		return 0;
	}
	return a;
}

double ReLU::differential(double a) {
	if (a < 0) {
		return 0;
	}
	return 1;
}

LReLU::LReLU(double leak): leak_(leak){}
LReLU::~LReLU(){}

double LReLU::activation(double a) {
	if (a < 0) {
		return leak_ * a;
	}
	return a;
}

double LReLU::differential(double a) {
	if (a < 0) {
		return leak_;
	}
	return 1;
}

double Sigmoid::activation(double a) {
	return 1 / (1 + exp(-a));
}

double Sigmoid::differential(double a) {
	return activation(a) * (1 - activation(a));
}

double LTwo::activation(double a) {
	return (a - sin(a)) * (a - sin(a));
}

double LTwo::differential(double a) {
	return a - sin(a);
}