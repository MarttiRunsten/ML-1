#pragma once

#include <vector>
#include <utility>
#include <random>
#include <memory>
#include <iostream> // For debug

#include "activations.h"

class Matrix;

using mPtr = std::shared_ptr<Matrix>;

class Matrix
{
public:
    Matrix(int h, int w, char type = 'r');
    Matrix(const mPtr ptr);
    Matrix(std::vector<std::vector<double>>& c);
    Matrix();
    ~Matrix();

    mPtr transpose();

    mPtr eWise(double(*f)(double));
    mPtr eWise(Base* b, bool dif = false);
    mPtr eWiseBin(BinBase* b, double leak, bool dif = false);
    mPtr eWiseMul(mPtr M);

    mPtr appendOne();
    mPtr removeCol();

    mPtr operator *(Matrix& M);
    mPtr operator *(int a);
    mPtr operator *(double a);

    mPtr operator +(Matrix& M);
    mPtr operator -(Matrix& M);

    double eSum();

    double at(int i, int j);
    void insert(int i, int j, double val);
    void clear();

    mPtr row(int i);
    mPtr col(int j);

    const std::pair<int, int> size();

    void print();

private:
    int h_;
    int w_;
    std::vector<std::vector<double>> contents_;
};
