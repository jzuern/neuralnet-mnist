#ifndef NEURALNET_H
#define NEURALNET_H

#include <Eigen/Dense> // eigen library for matrix vector stuff
#include <random>
#include "util.h"
#include <iostream>
#include <algorithm>
#include "Data.h"
#include <numeric>

using namespace Eigen;

class NeuralNet {

public:
        NeuralNet (int, int, int); // class constructor
        ~NeuralNet();              // class destructor

        void train( Data trainingData,const size_t nEpochs,const size_t mini_batch_size,const double learningRate,const double lambda,const Data validationData);

        VectorXd feed_forward(const VectorXd input); // forwardfeed input through neural net

        VectorXd cost_derivative_quad(const VectorXd output_activations,const VectorXd digitVec,const VectorXd z); // quadratic cost function

        VectorXd cost_derivative_cross_entropy(const VectorXd output_activations,const VectorXd y,const VectorXd z); // cross-entropy cost function

        void update(const std::vector<VectorXd> images,const std::vector<int> digits,const int idx,const double learningRate, const double lambda,const size_t mini_batch_size, const size_t dataSetSize);  // Update the network's weights and biases by applying gradient descent using backpropagation to a single mini batch.

        int eval( Data);  // evaluate the number of test inputs for which the neural network outputs the correct result.

        std::pair<std::vector<MatrixXd>,std::vector<VectorXd>> backprop(const VectorXd img,const int digit); // perform backpropagation algorithm with a single input image



private:
        size_t m_nInput; // number of input neurons
        size_t m_nHidden;// number of hidden neurons
        size_t m_nOutput;// number of output neurons
        std::vector<size_t> dims; // vector containing number of neurons in each layer
        size_t m_nLayers = 3; // our neural net has three layers (input layer, hidden layer, and output layer)
        std::vector<VectorXd> m_biases;  // every entry of m_biases stores the bias vector for a specific layer
        std::vector<MatrixXd> m_weights; // every entry of m_weights stores the weight matrix for a specific layer

};


#endif
