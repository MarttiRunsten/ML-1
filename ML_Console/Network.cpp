#include "network.h"

Layer::Layer(Network* parent, int input_size, int layer_size , Base* activation):
	net_(parent), isize_(input_size) , lsize_(layer_size), activ_(activation) {
	Matrix W(lsize_, isize_ + 1);
	Matrix A(lsize_, 1, 'o');
	Matrix O(A);
	Matrix D(A);
	Matrix I(isize_, 1, 'o');
	W_ = W;
	A_ = A;
	O_ = O;
	D_ = D;
	I_ = I;
}

Layer::~Layer(){}

void Layer::feedForward(Matrix& In) {
	Matrix Inb = In.appendOne(); // For the bias in matrix W_
	O_ = ((W_ * Inb).eWise(activ_));
	if (next_ != nullptr) {
		next_->feedForward(O_);
	}
	net_->receiveOutput(O_);
}

void Layer::backpropagate(Matrix& D_upper, Matrix& W_upper) {
	if (next_ == nullptr) {
		D_ = D_upper.eWiseMul(O_);
	}
	else {
		Matrix Fp = A_.eWise(activ_, true);
		D_ = ((W_upper.transpose()) * D_upper).eWiseMul(Fp);
	}
	updateW();
	if (prev_ != nullptr) {
		prev_->backpropagate(D_, W_);
	}
	else {
		net_->bpDone();
	}
}

std::pair<int, int> Layer::size() {
	std::pair<int, int> s(isize_, lsize_);
	return s;
}

/*
	Not at all compatible with batch learning. For that, all these values need to be averaged and
	all vectors (like inputs, outputs and deltas) will be matrices. Look into indexing before attempting!
*/
void Layer::updateW() {
	Matrix dW(lsize_, isize_ + 1, 'o');
	for (int i = 0; i < lsize_; i++) {
		dW.insert(i, 0, -(net_->getL()) * D_.at(i, 0)); // Bias adjustment (NOT BATCH COMPATIBLE)
		for (int j = 1; j < isize_ + 1; j++) {
			dW.insert(i, j, -(net_->getL() * I_.at(j, 0) * D_.at(i, 0) + net_->getR() * W_.at(i, j)));
		}
	}
	W_ + dW; // Updating values
}

Network::Network() {

}

Network::~Network() {}

void Network::feedInput() {

}

void Network::receiveOutput(Matrix& O) {

}

void Network::backpropagate() {

}

void Network::bpDone() {

}

double Network::getL() {
	return lambda_;
}

double Network::getR() {
	return rho_;
}
