#include "network.h"

Layer::Layer(Network* parent): net_(parent) {
	std::cout << "Layer constructor\n";
	W_ = nullptr;
	A_ = nullptr;
	O_ = nullptr;
	D_ = nullptr;
	I_ = nullptr;
	isize_ = 0;
	lsize_ = 0;
	next_ = nullptr;
	activ_ = nullptr;

	std::cout << "Setting prev_\n";
	prev_ = net_->getOut();
	if (prev_ != nullptr) {
		std::cout << " and prev_->next_\n";
		prev_->setNext(this);
	}
	std::cout << "Constructed\n";
}

Layer::~Layer() {
	delete W_;
	delete A_;
	delete O_;
	delete D_;
	delete I_;
}

void Layer::setup() {
	std::cout << "Layer::setup\n";
	if (prev_ == nullptr) {
		std::cout << "Set input size: ";
		std::cin >> isize_;
	}
	else {
		isize_ = prev_->size().second;
	}
	std::cout << "Set layer size: ";
	std::cin >> lsize_;

	W_ = new Matrix(lsize_, isize_ + 1);

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

void Layer::feedForward(Matrix* In) {
	std::cout << "Layer::feedForward\n";
	delete I_;
	I_ = In->appendOne(); // For the bias in matrix W_
	std::cout << "In:\n";
	In->print();
	std::cout << "moi\n";
	delete A_;
	A_ = *W_ * *I_;
	std::cout << "A_ (size " << I_->size().first << "x" << A_->size().second << "):\n";
	A_->print();
	delete O_;
	O_ = A_->eWise(activ_);
	std::cout << "O_:\n";
	O_->print();
	if (next_ != nullptr) {
		next_->feedForward(O_);
	}
	net_->receiveOutput(O_);
}

void Layer::backpropagate(Matrix* D_upper, Matrix* W_upper) {
	if (next_ == nullptr) {
		delete D_;
		D_ = D_upper->eWiseMul(O_);
	}
	else {
		Matrix* W_t = W_upper->transpose();
		Matrix* P = *W_t * *D_upper;
		Matrix* dO = A_->eWise(activ_, true);

		delete D_;
		D_ = P->eWiseMul(dO);

		delete W_t;
		delete P;
		delete dO;
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
	Matrix* dW = new Matrix(lsize_, isize_ + 1, 'o');
	for (int i = 0; i < lsize_; i++) {
		dW->insert(i, 0, -(net_->getL()) * D_->at(i, 0)); // Bias adjustment (NOT BATCH COMPATIBLE)
		for (int j = 1; j < isize_ + 1; j++) {
			dW->insert(i, j, -(net_->getL() * I_->at(j, 0) * D_->at(i, 0) + net_->getR() * W_->at(i, j)));
		}
	}
	delete W_;
	W_ = *W_ + *dW;
	delete dW;
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
	std::cout << "Network constructor\n";
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

		Layer* layer = nullptr;
		switch (choice)
		{
		case 1:
			layer = new Layer(this);
			layer->setup();
			if (in_ == nullptr) {
				in_ = layer;
			}
			out_ = layer;
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
	test();
}

Network::~Network() {}

void Network::feedInput() {
	Matrix* I = new Matrix(in_->size().first, 1);
	std::cout << "Input (size " << in_->size().first << "x1):\n";
	I->print();
	in_->feedForward(I);
	delete I;
}

void Network::receiveOutput(Matrix* O) { // set to L2, fitting sine
	Matrix* NablaL = O->eWise(loss_, true);
	out_->backpropagate(NablaL, O);
	delete NablaL;
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

Layer* Network::getOut() {
	std::cout << "Network::getOut\n";
	return out_;
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

void Network::test() {
	bool testing = true;
	int choice = 0;
	while (testing) {
		std::cout << "Testing network.\n[1] Run a forward-back loop\n[2] Quit\n";
		std::cin >> choice;
		if (choice == 1) {
			feedInput();
		}
		else if (choice == 2) {
			return;
		}
	}
}