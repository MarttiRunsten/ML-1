#pragma once

#include <cmath>
#include <iostream>


struct Base {
	virtual double activation(double a);
	virtual double differential(double a);
};

struct BinBase :public Base {
	virtual double activation(double a, double b);
	virtual double differential(double a, double b);
};

struct ReLU : public Base {
	double activation(double a) override;
	double differential(double a) override;
};

struct LReLU :public BinBase {
	double activation(double a, double b) override;
	double differential(double a, double b) override;
};

struct Sigmoid : public Base {
	double activation(double a) override;
	double differential(double a) override;
};
