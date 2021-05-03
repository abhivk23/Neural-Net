#ifndef NEURON_H
#define NEURON_H

#include <eigen3/Eigen/Dense>
#include <vector>
#include <fstream>
using std::vector;

class Neuron {   
    public:
        Neuron(Eigen::VectorXf w, float (*tf)(float, bool));

        int get_dim(void);
        Eigen::VectorXf get_weights(void);
        void set_weights(Eigen::VectorXf w);
        void set_tf(float (*tf)(float, bool));
        void slp_batch_train(Eigen::MatrixXf train_set, float alpha, int max_epoch, std::ofstream* training_log);
        Eigen::VectorXf slp_test(Eigen::MatrixXf test_set);
    private:
        float (*transfer_f)(float, bool);
        Eigen::VectorXf weights;
};

class Layer {
    public:
        Layer(vector<Neuron> array);
    private:
        vector<Neuron> neural_array;
};

class Network {
    public:
        Network(vector<Layer> net);
    private:
        vector<Layer> neural_net;
};

// void train(Eigen::MatrixXd train_set, Eigen::VectorXi output, float learningRate, int epochs);
float act_sigmoid(float arg, bool df);
float act_tanh(float arg, bool df);

#endif
