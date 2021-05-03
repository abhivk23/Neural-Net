// Abhiram Kakuturu
#include "neuron.hpp"
//#include <cmath>
#include <iostream>
using std::cout;
using std::endl;

// MISC Functions
Eigen::MatrixXf cwiseMult(Eigen::MatrixXf A, Eigen::MatrixXf B) {
    // if (A.size() != B.size()) return MatrixXf::Zero(A.rows(), A.cols());
    Eigen::MatrixXf C(A.rows(), A.cols());
    for(int r=0; r<A.rows(); r++) {
        for(int c=0; c<A.cols(); c++) C(r,c) = A(r, c) * B(r, c);
    }
    return C;
}

// if calculating first derivative pass true for
float act_sigmoid(float arg, bool Df){
    float sig = 1.0/(1+exp(-arg));
    if(Df) return sig*(1-sig);
    else return sig;
}

float act_tanh(float arg, bool Df) {
    float tan_h = tanh(arg);
    if(Df) return 1-(tan_h*tan_h);
    else return tan_h;
}

// NEURON MEMEBER FUNCTIONS
Neuron::Neuron(Eigen::VectorXf w, float (*tf)(float, bool)) {
    weights = w;
    set_tf(tf);
    return;
}

int Neuron::get_dim(void) { return weights.rows(); } 
Eigen::VectorXf Neuron::get_weights(void){ return weights;}
void Neuron::set_weights(Eigen::VectorXf w) {weights = w; return; }
void Neuron::set_tf(float (*tf)(float, bool)) { transfer_f = tf; return;}

void Neuron::slp_batch_train(Eigen::MatrixXf train_set, float alpha, int max_epoch, std::ofstream* training_log){
    // Preprocessing steps
    int item_count = train_set.rows(), component_count = train_set.cols()-1;
    if (component_count!=get_dim()){
        cout << "Improper number of inputs. Expected " << get_dim() << " but training set includes " << component_count << endl;
        return;
    }
    Eigen::MatrixXf o_i = train_set.block(0,0, item_count, component_count); // training data input
    Eigen::VectorXf t_j = train_set.block(0,component_count, item_count, 1); // training data output ('true' output to be tested against)
    
    // Train continuously for specified number of epochs
    for(int epoch=0; epoch<max_epoch; epoch++){
        Eigen::VectorXf o_j = o_i * weights; // apply weights to inputs
        for(int i=0; i<item_count; i++) o_j(i) = transfer_f(o_j(i), false); // forward propogation using specified activation function
        
        // Calculate error as difference between true and generated output
        Eigen::VectorXf error = o_j-t_j;

        // Calculate 'delta' from 
        Eigen::VectorXf first_deriv_o_j = o_j;
        for(int i=0; i<item_count; i++) first_deriv_o_j(i) = transfer_f(o_j(i), true); // activation function first derivative
        Eigen::VectorXf delta = cwiseMult(error, first_deriv_o_j);

        // Update neuron weights
        Eigen::VectorXf change = alpha*o_i.transpose()*delta;
        set_weights(weights-change);

        // Record absolute error by epoch in training log
        *training_log << epoch << "," << error.cwiseAbs().sum() << endl;
    }
    return;
}

Eigen::VectorXf Neuron::slp_test(Eigen::MatrixXf test_set) {
    int item_count = test_set.rows(), component_count = test_set.cols()-1;
    if (component_count!=get_dim()) return test_set.block(0,0,0,1); // and throw exception
    Eigen::MatrixXf o_i = test_set.block(0,0, item_count, component_count);
    //Eigen::VectorXf t_j = test_set.block(0,component_count, item_count, 1);

    Eigen::VectorXf o_j = o_i * weights;
    for(int i=0; i<item_count; i++) o_j(i) = transfer_f(o_j(i), false);
    return o_j;

}

// LAYER MEMBER FUNCTIONS
Layer::Layer(vector<Neuron> array) {
    neural_array = array;
}

// NETWORK MEMBER FUNCTIONS
Network::Network(vector<Layer> net) {
    neural_net = net;
    return;
}