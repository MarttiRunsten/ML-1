#include "network.h"

Layer::Layer(Network* parent, int input_size, int layer_size , Base* activation):
	net_(parent), isize_(input_size) , lsize_(layer_size), activ_(activation) {
	Matrix W(lsize_, isize_ + 1);
	Matrix A(lsize_, 1, 'o');
	Matrix O(A);
	Matrix D(A);
	W_ = W;
	A_ = A;
	O_ = O;
	D_ = D;
}

Layer::~Layer(){}

void Layer::feedForward(Matrix& In) {
	Matrix Inb = In.appendOne();
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

void Layer::updateW() {

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
