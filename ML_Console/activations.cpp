#include "activations.h"

double Base::activation(double a) {
	return a;
}

double Base::differential(double a) {
	return 0;
}

double BinBase::activation(double a, double b) {
	return a;
}

double BinBase::differential(double a, double b) {
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

double LReLU::activation(double a, double b) {
	if (a < 0) {
		return b * a;
	}
	return a;
}

double LReLU::differential(double a, double b) {
	if (a < 0) {
		return b;
	}
	return 1;
}

double Sigmoid::activation(double a) {
	return 1 / (1 + exp(-a));
}

double Sigmoid::differential(double a) {
	return activation(a) * (1 - activation(a));
}