#include "matrix.h"

Matrix::Matrix(int h, int w, char type) : h_(h), w_(w) {
    std::vector<std::vector<double>> c;
    switch (type) {
    case 'o':
        for (int i = 0; i < h_; i++) {
            std::vector<double> row(w_, 0);
            c.push_back(row);
        }
        break;
    case 'i':
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
        std::random_device rd;
        std::mt19937 g(rd());
        std::uniform_real_distribution<> dis(-10, 10);
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

Matrix::Matrix(const mPtr ptr) {
    h_ = ptr->h_;
    w_ = ptr->w_;
    contents_ = ptr->contents_;
}

Matrix::Matrix(std::vector<std::vector<double>>& c) {
    h_ = c.size();
    w_ = c.front().size();
    contents_ = c;
}

Matrix::Matrix() {
    h_ = 1;
    w_ = 1;
    std::vector<double> cell(1, 1);
    contents_.push_back(cell);
}

Matrix::~Matrix() {}

mPtr Matrix::transpose() {
    mPtr R = std::make_shared<Matrix>(Matrix(w_, h_, 'o'));
    for (int i = 0; i < h_; i++) {
        for (int j = 0; j < w_; j++) {
            R->insert(j, i, at(i, j));
        }
    }
    return R;
}

mPtr Matrix::eWise(double(*f)(double)) {
    mPtr R = std::make_shared<Matrix>(Matrix(h_, w_, 'o'));
    for (int i = 0; i < h_; i++) {
        for (int j = 0; j < w_; j++) {
            R->insert(i, j, f(at(i, j)));
        }
    }
    return R;
}

mPtr Matrix::eWise(Base* b, bool dif) {
    mPtr R = std::make_shared<Matrix>(Matrix(h_, w_, 'o'));
    for (int i = 0; i < h_; i++) {
        for (int j = 0; j < w_; j++) {
            if (dif) {
                R->insert(i, j, b->differential(at(i, j)));
            }
            else {
                R->insert(i, j, b->activation(at(i, j)));
            }
        }
    }
    return R;
}

mPtr Matrix::eWiseBin(BinBase* b, double leak, bool dif) {
    mPtr R = std::make_shared<Matrix>(Matrix(h_, w_, 'o'));
    for (int i = 0; i < h_; i++) {
        for (int j = 0; j < w_; j++) {
            if (dif) {
                R->insert(i, j, b->differential(at(i, j), leak));
            }
            else {
                R->insert(i, j, b->activation(at(i, j), leak));
            }
        }
    }
    return R;
}

mPtr Matrix::eWiseMul(mPtr M) {
    mPtr R = std::make_shared<Matrix>(Matrix(h_, w_, 'o'));
    for (int i = 0; i < h_; i++) {
        for (int j = 0; j < w_; j++) {
            R->insert(i, j, (at(i, j)) * (M->at(i, j)));
        }
    }
    return R;
}

mPtr Matrix::appendOne() {
    mPtr R = std::make_shared<Matrix>(Matrix(h_ + 1, 1, 'o'));
    R->insert(0, 0, 1);
    for (int i = 0; i < h_; i++) {
        R->insert(i + 1, 0, at(i, 0));
    }
    return R;
}

mPtr Matrix::removeCol() {
    mPtr R = std::make_shared<Matrix>(Matrix(h_, w_ - 1, 'o'));
    for (int i = 0; i < h_; i++) {
        for (int j = 1; j < w_; j++) {
            R->insert(i, j - 1, at(i, j));
        }
    }
    return R;
}

mPtr Matrix::operator *(Matrix& M) {
    if (size().second != M.size().first) {
        return nullptr;
    }
    mPtr R = std::make_shared<Matrix>(Matrix(size().first, M.size().second, 'o'));
    double s;
    for (int i = 0; i < R->size().first; i++) {
        for (int j = 0; j < R->size().second; j++) {
            s = 0;
            for (int k = 0; k < size().second; k++) {
                s += at(i, k) * M.at(k, j);
            }
            R->insert(i, j, s);
        }
    }
    return R;
}

mPtr Matrix::operator *(int a) {
    mPtr R = std::make_shared<Matrix>(Matrix(h_, w_, 'o'));
    for (int i = 0; i < h_; i++) {
        for (int j = 0; j < w_; j++) {
            R->insert(i, j, at(i, j) * a);
        }
    }
    return R;
}

mPtr Matrix::operator *(const double a) {
    mPtr R = std::make_shared<Matrix>(Matrix(h_, w_, 'o'));
    for (int i = 0; i < h_; i++) {
        for (int j = 0; j < w_; j++) {
            R->insert(i, j, at(i, j) * a);
        }
    }
    return R;
}

mPtr Matrix::operator +(Matrix& M) {
    mPtr R = std::make_shared<Matrix>(Matrix(h_, w_, 'o'));
    for (int i = 0; i < h_; i++) {
        for (int j = 0; j < w_; j++) {
            R->insert(i, j, at(i, j) + M.at(i, j));
        }
    }
    return R;
}

mPtr Matrix::operator -(Matrix& M) {
    mPtr R = std::make_shared<Matrix>(Matrix(h_, w_, 'o'));
    for (int i = 0; i < h_; i++) {
        for (int j = 0; j < w_; j++) {
            R->insert(i, j, at(i, j) - M.at(i, j));
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

void Matrix::clear() {
    for (int i = 0; i < h_; i++) {
        for (int j = 0; j < w_; j++) {
            insert(i, j, 0);
        }
    }
}

mPtr Matrix::row(int i) {
    mPtr r = std::make_shared<Matrix>(Matrix(1, w_, 'o'));
    for (int j = 0; j < w_; j++) {
        r->insert(1, j, at(i, j));
    }
    return r;
}

mPtr Matrix::col(int j) {
    mPtr c = std::make_shared<Matrix>(Matrix(h_, 1, 'o'));
    for (int i = 0; i < h_; i++) {
        c->insert(i, 1, at(i, j));
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
