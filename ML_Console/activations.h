#pragma once

#include <cmath>

struct Base {
	virtual double activation(double a);
	virtual double differential(double a);
};

struct ReLU : public Base {
	double activation(double a) override;
	double differential(double a) override;
};

struct LReLU :public Base {
	double activation(double a) override;
	double differential(double a) override;
};

struct Sigmoid : public Base {
	double activation(double a) override;
	double differential(double a) override;
};
