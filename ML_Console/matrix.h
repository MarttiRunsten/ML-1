#pragma once

#include <vector>
#include <utility>
#include <random>
#include <iostream> // For debug

#include "activations.h"

class Matrix
{
public:
    Matrix(int h, int w, char type = 'r');
    Matrix(const Matrix& M);
    Matrix(std::vector<std::vector<double>>& c);
    Matrix();
    ~Matrix();

    Matrix transpose();

    Matrix eWise(double(*f)(double));
    Matrix eWise(Base* b, bool dif = false);

    Matrix eWiseMul(Matrix& M);

    Matrix appendOne();

    Matrix operator *(Matrix& M);
    Matrix operator *(int a);
    Matrix operator *(double a);

    Matrix operator +(Matrix& M);
    Matrix operator -(Matrix& M);

    double eSum();

    double at(int i, int j);
    void insert(int i, int j, double val);
    void clear();

    Matrix row(int i);
    Matrix col(int j);

    const std::pair<int, int> size();

    void print();

private:
    int h_;
    int w_;
    std::vector<std::vector<double>> contents_;
};
