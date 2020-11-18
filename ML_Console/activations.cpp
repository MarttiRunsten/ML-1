#include "activations.h"

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

double LReLU::activation(double a) {
	if (a < 0) {
		return .1 * a;
	}
	return a;
}

double LReLU::differential(double a) {
	if (a < 0) {
		return .1;
	}
	return 1;
}

double Sigmoid::activation(double a) {
	return 1 / (1 + exp(-a));
}

double Sigmoid::differential(double a) {
	return activation(a) * (1 - activation(a));
}