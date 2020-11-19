#include "network.h"

Layer::Layer(Network* parent): net_(parent) {
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
	isize_ = 0;
	lsize_ = 0;
	next_ = nullptr;
	prev_ = nullptr;
}

Layer::~Layer(){}

void Layer::setup() {
	if (prev_ == nullptr) {
		std::cout << "Set input size: ";
		std::cin >> isize_;
	}
	else {
		isize_ = prev_->size().second;
	}
	std::cout << "Set layer size: ";
	std::cin >> lsize_;

	int choice = 0;
	bool choosing = true;
	while (choosing) {
		std::cout << "Choose activation:\n"
			<< "[1] ReLU\n"
			<< "[2] LReLU\n"
			<< "[3] Sigmoid\n";
		std::cin >> choice;
		switch (choice)
		{
		case 1:
			makeRelu();
			break;
		case 2:
			makeLrelu();
			break;
		case 3:
			makeSigmoid();
			break;
		default:
			break;
		}
		if (activ_ != nullptr) {
			break;
		}
	}
}

void Layer::setNext(Layer* n) {
	next_ = n;
}

void Layer::setPrev(Layer* p) {
	prev_ = p;
}

void Layer::feedForward(Matrix& In) {
	I_ = In.appendOne(); // For the bias in matrix W_
	A_ = W_ * I_;
	O_ = A_.eWise(activ_);
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

void Layer::makeRelu() {
	ReLU a;
	activ_ = &a;
}

void Layer::makeLrelu() {
	double l;
	std::cout << "Choose leak strength (0 to 1): ";
	std::cin >> l;
	LReLU a(l);
	activ_ = &a;
}

void Layer::makeSigmoid() {
	Sigmoid a;
	activ_ = &a;
}

Network::Network() {
	in_ = nullptr;
	out_ = nullptr;
	loss_ = nullptr;
	train_ = 0;
	iter_ = 0;

	setHypers();

	bool setup = true;
	int choice;
	do {
		choice = 0;
		std::cout	<< "[1] Create layer\n"
					<< "[2] set lambda/rho\n"
					<< "[3] Start testing\n";
		std::cin >> choice;

		Layer l(this);
		switch (choice)
		{
		case 1:
			l.setup();
			if (in_ == nullptr) {
				in_ = &l;
			}
			out_ = &l;
			break;
		case 2:
			setHypers();
			break;
		case 3:
			if (in_ == nullptr) {
				std::cout << "You must add at least 1 layer!\n";
				break;
			}
			setup = false;
			break;
		default:
			std::cout << "Unknown choice.\n";
			break;
		}
	} while (setup);
}

Network::~Network() {}

void Network::feedInput() {
	Matrix I(in_->size().first, 1);
	in_->feedForward(I);
}

void Network::receiveOutput(Matrix& O) { // set to L2, fitting sine
	Matrix NablaL(O.eWise(loss_, true));
	out_->backpropagate(NablaL, O);
}

void Network::bpDone() {
	if (train_ > 0) {
		feedInput();
	}
}

double Network::getL() {
	return lambda_;
}

double Network::getR() {
	return rho_;
}

void Network::setHypers() {
	std::cout << "Set learning rate: ";
	std::cin >> lambda_;

	std::cout << "Set regularization parameter: ";
	std::cin >> rho_;
}

void Network::makeLoss() {
	LTwo l;
	loss_ = &l;
}