// Abhiram Kakuturu
// C++14
#include "neuron.hpp"
#include <iostream>
#include <time.h>
using std::cout;
using std::endl;


// Learning Rate analysis across tanh and sigmoid activation functions
int main(void){
    // Initialize neurons with equal, normalized weightage on each input
    Eigen::VectorXf norm_weights(3,1);
    float norm_w = 1.0/norm_weights.size();
    for(int i=0; i<norm_weights.size(); i++) norm_weights(i)=norm_w;

    // OR Initialize neurons with equal, ZEROED weightage on each input
    Eigen::VectorXf null_weights(3,1);
    for(int i=0; i<null_weights.size(); i++) null_weights(i)=0;

    // Learning variables 
    float learning_rate[] = {100, 1.0, 0.1};
    int max_epoch = 10001;

    // Training and Testing data
    Eigen::Matrix<float, 4,4> train_data;
    train_data << 0.0, 0.0, 1.0, 0.0,
                  1.0, 1.0, 1.0, 1.0,
                  1.0, 0.0, 1.0, 1.0,
                  0.0, 1.0, 1.0, 1.0;
    Eigen::Matrix<float, 4,4> test_data;
    test_data << 0.0, 0.0, 0.0, 0.0,
                 1.0, 1.0, 0.0, 1.0,
                 1.0, 0.0, 0.0, 1.0,
                 0.0, 1.0, 0.0, 1.0;

    int i=0;
    for(float alpha : learning_rate){
        // CSV training logs for keeping track of error rate across epoch
        std::ofstream training_log_sig("../Output/sigmoid_output_"+std::to_string(i)+".txt");
        std::ofstream training_log_tanh("../Output/tanh_output_"+std::to_string(i)+".txt");
        training_log_sig << "Learning Rate: " << alpha << endl << "EPOCH,ABS_ERROR" << endl;
        training_log_tanh << "Learning Rate: " << alpha << endl << "EPOCH,ABS_ERROR" << endl;

        // Initialize neurons with impartial weights and desired activation functions
        Neuron n_sig(null_weights, act_sigmoid);
        Neuron n_tanh(null_weights, act_tanh);

        // Train neurons and print training speeds to console
        int s1 = clock();
        n_sig.slp_batch_train(train_data, alpha, max_epoch, &training_log_sig);
        cout << clock() - s1 << " ticks for sigmoid." << endl;
        int s2 = clock();
        n_tanh.slp_batch_train(train_data, alpha, max_epoch, &training_log_tanh);
        cout << clock() - s2 << " ticks for tanh." << endl;
        //cout << "Trained Weights Sigmoid Neuron: " << endl << n_sig.get_weights() << endl << endl;
        //cout << "Trained Weights Tanh Neuron: " << endl << n_tanh.get_weights() << endl << endl;

        cout << "RUN #" << i+1 << " with learning rate alpha=" << alpha << endl;
        cout << "Correct output: " << test_data.block(0,test_data.cols()-1,test_data.rows(),1).transpose() << endl;
        cout << "Sigmoid Neuron output: " << n_sig.slp_test(test_data).transpose() << endl;
        cout << "Tanh Neuron output: " << n_tanh.slp_test(test_data).transpose() << endl;
        i++;
    }

    return 0;
}
