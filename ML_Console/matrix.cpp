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

Matrix::Matrix(const Matrix* ptr) {
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

Matrix* Matrix::transpose() {
    Matrix* R = new Matrix(w_, h_, 'o');
    for (int i = 0; i < h_; i++) {
        for (int j = 0; j < w_; j++) {
            R->insert(j, i, at(i, j));
        }
    }
    return R;
}

Matrix* Matrix::eWise(double(*f)(double)) {
    Matrix* R = new Matrix(h_, w_, 'o');
    for (int i = 0; i < h_; i++) {
        for (int j = 0; j < w_; j++) {
            R->insert(i, j, f(at(i, j)));
        }
    }
    return R;
}

Matrix* Matrix::eWise(Base* b, bool dif) {
    std::cout << "Matrix::eWise ";
    Matrix* R = new Matrix(h_, w_, 'o');
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
    std::cout << "done\n";
    return R;
}

Matrix* Matrix::eWiseBin(BinBase* b, double leak, bool dif) {
    std::cout << "Matrix::eWiseBin ";
    Matrix* R = new Matrix(h_, w_, 'o');
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
    std::cout << "done\n";
    return R;
}

Matrix* Matrix::eWiseMul(Matrix* M) {
    std::cout << "Matrix::eWiseMul " << h_ << "x" << w_ << " with " << M->h_ << "x" << M->w_ << '\n';
    Matrix* R = new Matrix(h_, w_, 'o');
    for (int i = 0; i < h_; i++) {
        for (int j = 0; j < w_; j++) {
            R->insert(i, j, (at(i, j)) * (M->at(i, j)));
        }
    }
    std::cout << "done\n";
    return R;
}

Matrix* Matrix::appendOne() {
    Matrix* R = new Matrix(h_ + 1, 1, 'o');
    R->insert(0, 0, 1);
    for (int i = 0; i < h_; i++) {
        R->insert(i + 1, 0, at(i, 0));
    }
    return R;
}

Matrix* Matrix::removeCol() {
    std::cout << "Matrix::removeCol ";
    Matrix* R = new Matrix(h_, w_ - 1, 'o');
    for (int i = 0; i < h_; i++) {
        for (int j = 1; j < w_; j++) {
            R->insert(i, j - 1, at(i, j));
        }
    }
    std::cout << "done\n";
    return R;
}

Matrix* Matrix::operator *(Matrix& M) {
    std::cout << "Operator *, multiplying (" << h_ << "x" << w_ << ") to (" << M.h_ << "x" << M.w_ << ")\n";
    if (size().second != M.size().first) {
        return nullptr;
    }
    Matrix* R = new Matrix(size().first, M.size().second, 'o');
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
    std::cout << "done\n";
    return R;
}

Matrix* Matrix::operator *(int a) {
    Matrix* R = new Matrix(h_, w_, 'o');
    for (int i = 0; i < h_; i++) {
        for (int j = 0; j < w_; j++) {
            R->insert(i, j, at(i, j) * a);
        }
    }
    return R;
}

Matrix* Matrix::operator *(const double a) {
    Matrix* R = new Matrix(h_, w_, 'o');
    for (int i = 0; i < h_; i++) {
        for (int j = 0; j < w_; j++) {
            R->insert(i, j, at(i, j) * a);
        }
    }
    return R;
}

Matrix* Matrix::operator +(Matrix& M) {
    Matrix* R = new Matrix(h_, w_, 'o');
    for (int i = 0; i < h_; i++) {
        for (int j = 0; j < w_; j++) {
            R->insert(i, j, at(i, j) + M.at(i, j));
        }
    }
    return R;
}

Matrix* Matrix::operator -(Matrix& M) {
    Matrix* R = new Matrix(h_, w_, 'o');
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

Matrix* Matrix::row(int i) {
    Matrix* r = new Matrix(1, w_, 'o');
    for (int j = 0; j < w_; j++) {
        r->insert(1, j, at(i, j));
    }
    return r;
}

Matrix* Matrix::col(int j) {
    Matrix* c = new Matrix(h_, 1, 'o');
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
