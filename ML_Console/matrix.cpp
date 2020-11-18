#include "matrix.h"

Matrix::Matrix(int h, int w, char type) : h_(h), w_(w) {
    std::cout << "Constructor, ";
    std::vector<std::vector<double>> c;
    switch (type) {
    case 'm':
        break;
    case 'o':
        std::cout << "zeros...\n";
        for (int i = 0; i < h_; i++) {
            std::vector<double> row(w_, 0);
            c.push_back(row);
        }
        break;
    case 'i':
        std::cout << "identity...\n";
        for (int i = 0; i < h_; i++) {
            std::vector<double> row;
            for (int j = 0; j < w_; j++) {
                if (i == j) {
                    row.push_back(1);
                }
                else {
                    row.push_back(0);
                }
            }
            c.push_back(row);
        }
        break;
    default:
        std::cout << "random...\n";
        std::random_device rd;
        std::mt19937 g(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);
        for (int i = 0; i < h_; i++) {
            std::vector<double> row;
            for (int j = 0; j < w_; j++) {
                row.push_back(dis(g));
            }
            c.push_back(row);
        }
        break;
    }
    contents_ = c;
}

Matrix::Matrix(const Matrix& M) {
    std::cout << "Constructing copy...\n";
    h_ = M.h_;
    w_ = M.w_;
    contents_ = M.contents_;
}

Matrix::~Matrix() {}

Matrix Matrix::transpose() {
    std::cout << "Transposing...\n";
    Matrix R(w_, h_, 'o');
    for (int i = 0; i < h_; i++) {
        for (int j = 0; j < w_; j++) {
            R.insert(j, i, at(i, j));
        }
    }
    std::cout << "Transposing done!\n";
    return R;
}

Matrix Matrix::eWise(double(*f)(double)) {
    Matrix R(h_, w_, 'o');
    for (int i = 0; i < h_; i++) {
        for (int j = 0; j < w_; j++) {
            R.insert(i, j, f(at(i, j)));
        }
    }
    return R;
}

Matrix Matrix::eWiseMul(Matrix& M) {
    Matrix R(h_, w_, 'o');
    for (int i = 0; i < h_; i++) {
        for (int j = 0; j < w_; j++) {
            R.insert(i, j, (at(i, j)) * (M.at(i, j)));
        }
    }
    return R;
}

Matrix Matrix::operator *(Matrix& M) {
    std::cout << "Calculating product...\n";
    if (size().second != M.size().first) {
        Matrix F(1, 1, 'o');
        return F;
    }
    Matrix R(size().first, M.size().second, 'o');
    double s;
    for (int i = 0; i < R.size().first; i++) {
        for (int j = 0; j < R.size().second; j++) {
            s = 0;
            for (int k = 0; k < size().second; k++) {
                s += at(i, k) * M.at(k, j);
            }
            R.insert(i, j, s);
        }
    }
    std::cout << "Product ready!\n";
    return R;
}

Matrix Matrix::operator *(int a) {
    Matrix R(h_, w_, 'o');
    for (int i = 0; i < h_; i++) {
        for (int j = 0; j < w_; j++) {
            R.insert(i, j, at(i, j) * a);
        }
    }
    return R;
}

Matrix Matrix::operator *(const double a) {
    Matrix R(h_, w_, 'o');
    for (int i = 0; i < h_; i++) {
        for (int j = 0; j < w_; j++) {
            R.insert(i, j, at(i, j) * a);
        }
    }
    return R;
}

Matrix Matrix::operator +(Matrix& M) {
    Matrix R(h_, w_, 'o');
    for (int i = 0; i < h_; i++) {
        for (int j = 0; j < w_; j++) {
            R.insert(i, j, at(i, j) + M.at(i, j));
        }
    }
    return R;
}

Matrix Matrix::operator -(Matrix& M) {
    Matrix R(h_, w_, 'o');
    for (int i = 0; i < h_; i++) {
        for (int j = 0; j < w_; j++) {
            R.insert(i, j, at(i, j) - M.at(i, j));
        }
    }
    return R;
}

double Matrix::eSum() {
    double s = 0;
    for (int i = 0; i < h_; i++) {
        for (int j = 0; j < w_; j++) {
            s += at(i, j);
        }
    }
    return s;
}

double Matrix::at(int i, int j) {
    return contents_.at(i).at(j);
}

void Matrix::insert(int i, int j, double val) {
    contents_.at(i).at(j) = val;
}

Matrix Matrix::row(int i) {
    Matrix r(1, w_, 'o');
    for (int j = 0; j < w_; j++) {
        r.insert(1, j, at(i, j));
    }
    return r;
}

Matrix Matrix::col(int j) {
    Matrix c(h_, 1, 'o');
    for (int i = 0; i < h_; i++) {
        c.insert(i, 1, at(i, j));
    }
    return c;
}

const std::pair<int, int> Matrix::size() {
    std::pair<int, int> p(h_, w_);
    return p;
}

void Matrix::print() {
    for (int i = 0; i < h_; i++) {
        for (int j = 0; j < w_; j++) {
            std::cout << at(i, j) << " ";
        }
        std::cout << "\n\n";
    }
}