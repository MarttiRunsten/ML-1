#include "network.h"

Layer::Layer(Network* parent): net_(parent) {
	W_ = nullptr;
	A_ = nullptr;
	O_ = nullptr;
	D_ = nullptr;
	I_ = nullptr;
	isize_ = 0;
	lsize_ = 0;
	leak_ = 0;
	isBin_ = false;
	next_ = nullptr;
	activ_ = nullptr;

	prev_ = net_->getOut();
	if (prev_ != nullptr) {
		prev_->setNext(this);
	}
}

Layer::~Layer() {
	delete W_;
	delete A_;
	delete O_;
	delete D_;
	delete I_;
}

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
			activ_ = net_->RELU;
			break;
		case 2:
			std::cout << "Choose leak strength (0 to 1): ";
			std::cin >> leak_;
			isBin_ = true;
			activ_ = net_->LRELU;
			break;
		case 3:
			activ_ = net_->SIGMOID;
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
	delete In;
	delete A_;
	std::cout << W_->size().second << " = " << I_->size().first << std::endl;
	A_ = *W_ * *I_;
	delete O_;
	if (isBin_) {
		O_ = A_->eWiseBin(net_->LRELU, leak_);
	}
	else {
		O_ = A_->eWise(activ_);
	}
	if (next_ != nullptr) {
		std::cout << "Next layer...\n";
		next_->feedForward(O_);
	}
	std::cout << "Final answer :D\n";
	O_->print();
	net_->receiveOutput(O_);
}

void Layer::backpropagate(Matrix* D_upper, Matrix* W_upper) {
	std::cout << "Layer::backpropagate ";
	if (next_ == nullptr) {
		std::cout << "output layer\n";
		delete D_;
		D_ = D_upper->eWiseMul(O_);
	}
	else {
		std::cout << "inner layer\n";
		Matrix* W_pure = W_upper->removeCol();
		Matrix* W_t = W_pure->transpose();
		Matrix* P = *W_t * *D_upper; 
		Matrix* dO = nullptr;
		if (isBin_) {
			dO = A_->eWiseBin(net_->LRELU, leak_, true);
		}
		else {
			dO = A_->eWise(activ_, true);
		}

		delete D_;
		D_ = P->eWiseMul(dO);

		delete W_pure;
		delete W_t;
		delete P;
		delete dO;
	}
	updateW();
	if (prev_ != nullptr) {
		std::cout << "Next layer. Dims: D(" << D_->size().first << "x" << D_->size().second << "), W(" << W_->size().first << "x" << W_->size().second << ")\n";
		prev_->backpropagate(D_, W_);
	}
	else {
		std::cout << "Layer H1 bp done\n";
		net_->bpDone();
	}
}

std::pair<int, int> Layer::size() {
	std::pair<int, int> s(isize_, lsize_);
	return s;
}

Layer* Layer::getPrev() {
	return prev_;
}

Matrix* Layer::getI() {
	return I_;
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
	Matrix* ptr = W_;
	W_ = *W_ + *dW;
	delete ptr;
	std::cout << "W_ exists: \n";
	W_->print();
	std::cout << "dW:\n";
	dW->print();
	delete dW;
}

Network::Network() {
	RELU = new ReLU;
	LRELU = new LReLU;
	SIGMOID = new Sigmoid;

	std::cout << "Network constructor\n";
	in_ = nullptr;
	out_ = nullptr;
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

Network::~Network() {
	std::cout << "Deleting network... ";
	Layer* ptr = out_;
	while (ptr != nullptr) {
		ptr = out_->getPrev();
		delete out_;
	}

	delete RELU;
	delete LRELU;
	delete SIGMOID;
	std::cout << "done\n";
}

void Network::feedInput() {
	Matrix* I = new Matrix(in_->size().first, 1);
	std::cout << "Input (size " << in_->size().first << "x1):\n";
	I->print();
	in_->feedForward(I);
	delete I;
}

void Network::receiveOutput(Matrix* O) { // set to L2, fitting sine
	Matrix* NablaL = *O - *(in_->getI()->eWise(sin_));
	Matrix* losses = NablaL->eWiseMul(NablaL);
	std::cout << "Is this loss? (" << losses->eSum() << ")\n";
	delete losses;
	out_->backpropagate(NablaL, O);
	delete NablaL;
}

void Network::bpDone() {
	std::cout << "Network::bpDone\n";
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
	return out_;
}

void Network::setHypers() {
	std::cout << "Set learning rate: ";
	std::cin >> lambda_;

	std::cout << "Set regularization parameter: ";
	std::cin >> rho_;
}

void Network::test() {
	bool testing = true;
	int choice = 0;
	while (testing) {
		std::cout << "Testing network.\n[1] Run a forward-back loop\n[2] 3 test loops\n[Anything else] Quit\n";
		std::cin >> choice;
		if (choice == 1) {
			feedInput();
		}
		else if (choice == 2) {
			train_ = 3;
		}
		feedInput();
	}
}