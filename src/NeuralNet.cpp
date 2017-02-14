#include "NeuralNet.h"


NeuralNet::NeuralNet(int nInput, int nHidden, int nOutput) { // constructor of the NeuralNet class

        m_nInput = nInput;
        m_nHidden = nHidden;
        m_nOutput = nOutput;

        dims = {m_nInput, m_nHidden, m_nOutput};

        std::cout << "Neural Network Dimensions: ";
        std::cout << dims[0] << " " << dims[1] << " " << dims[2] << std::endl;


        // Initialize network with gaussian normal distribution, standard deviation of 1/sqrt(m_nInput)
        double stddev_biases = 1.0;
        double stddev_weights = 1.0/sqrt(m_nInput);

        std::default_random_engine generator; // (pseudo-)random number generator
        std::normal_distribution<double> distribution_biases(0.0,stddev_biases);
        std::normal_distribution<double> distribution_weights(0.0,stddev_weights);


        VectorXd biases_1 = VectorXd::Zero(m_nHidden);
        VectorXd biases_2 = VectorXd::Zero(m_nOutput);

        for (size_t i = 0; i < m_nHidden; i++){
          biases_1(i) = distribution_biases(generator); // vector of biases for the hidden layer
        }
        for (size_t i = 0; i < m_nOutput; i++){
          biases_1(i) = distribution_biases(generator); // vector of biases for the output layer
        }

        m_biases.push_back(biases_1);
        m_biases.push_back(biases_2);

        MatrixXd weights_1 = MatrixXd::Zero(m_nHidden,m_nInput);  // matrix of weights for connections between input layer and hidden layer
        MatrixXd weights_2 = MatrixXd::Zero(m_nOutput,m_nHidden); // matrix of weights for connections between hidden layer and output layer

        for (size_t i = 0; i < m_nHidden; i++){
          for (size_t j = 0; j < m_nInput; j++){
            weights_1(i,j) = distribution_weights(generator);
          }
        }

        for (size_t i = 0; i < m_nOutput; i++){
          for (size_t j = 0; j < m_nHidden; j++){
            weights_2(i,j) = distribution_weights(generator);
          }
        }

        m_weights.push_back(weights_1);
        m_weights.push_back(weights_2);


}


void NeuralNet::train( Data trainingData,const size_t nEpochs,const size_t mini_batch_size,const double learningRate,const double lambda,const Data validationData){

        // Train the neural network using mini-batch stochastic gradient descent.

        int nValSets = validationData.nEntries;

        std::vector<VectorXd> mini_batches_img;
        std::vector<int> mini_batches_digits;

        for(size_t j=0; j < nEpochs; j++) { // iterate through the training epochs

                std::cout << "Epoch " << j << " started..." <<  std::endl;
                size_t dataSetSize = 5000;

                // need random indices from 0 to dataSetSize-1
                std::vector<int> randomIndices;
                int max = dataSetSize-1-mini_batch_size; // must have mini_batch_size distance from last index
                int min = 0;
                for (size_t i = 0; i < dataSetSize; i++) {
                        randomIndices.push_back(rand()%(max-min + 1) + min);
                }

                mini_batches_img.clear();
                mini_batches_digits.clear();

                // populate mini_batches with random data entries
                for (size_t i = 0; i < dataSetSize; i++) {
                        mini_batches_img.push_back(trainingData.getImgEntry(randomIndices[i]));
                        mini_batches_digits.push_back(trainingData.getDigitEntry(randomIndices[i]));
                }


                for(size_t i = 0; i < dataSetSize; i+= mini_batch_size) {
                        if(i % 500 == 0) std::cout << "        " << i << " of " <<  dataSetSize << std::endl;
                        update(mini_batches_img , mini_batches_digits , i, learningRate , lambda , mini_batch_size, dataSetSize);
                }

                std::cout << "Epoch " << j << ": Success rate of " << 100.0 * (float)eval(validationData)/nValSets << " %"<<  std::endl;
        }

}

VectorXd NeuralNet::feed_forward(const VectorXd input){

      // feed input through network

      VectorXd in, out;
      in = input;

      for (size_t i = 0; i < m_nLayers-1; i++){ // feed input through the network layers
        out = m_weights[i]*in;
        out = sig(out + m_biases[i]);
        in = out;
      }

      return out;
}



VectorXd NeuralNet::cost_derivative_cross_entropy(const VectorXd output_activations,const VectorXd digitVec,const VectorXd z){ // cross-entropy cost function

        // partial derivative of cost function using cross-entropy cost function

        return output_activations - digitVec;
}

VectorXd NeuralNet::cost_derivative_quad( VectorXd output_activations, VectorXd digitVec, VectorXd z){ //  quadratic cost function

        // partial derivative of cost function using quadratic cost function

        VectorXd tmp = output_activations - digitVec;
        VectorXd delta = tmp.array() * sig_prime(z).array(); // conversion to array in order to perform element-by-element vector multiplication

        return delta;
}


void NeuralNet::update(const std::vector<VectorXd> images,const std::vector<int> digits,const int data_idx,\
  const double learningRate,const double lambda,const size_t mini_batch_size, const size_t dataSetSize){

        // Update the network's weights and biases by applying
        // gradient descent using backpropagation to a single mini batch.

        std::vector<MatrixXd> nabla_w;
        std::vector<VectorXd> nabla_b;

        // preallocate nabla_b and nabla_w with zeros
        nabla_w.push_back(MatrixXd::Zero(m_nHidden,m_nInput));
        nabla_b.push_back(VectorXd::Zero(m_nHidden));
        nabla_w.push_back(MatrixXd::Zero(m_nOutput,m_nHidden));
        nabla_b.push_back(VectorXd::Zero(m_nOutput));

        std::pair<std::vector<MatrixXd>, std::vector<VectorXd>> delta_nablas;

        // iterate through mini batch
        for (size_t offs = 0; offs < mini_batch_size; offs++) {


                int idx = data_idx + offs;

                delta_nablas = backprop(images[idx],digits[idx]);

                for (int i = 0; i < m_nLayers-1; i++) {
                  nabla_w[i] += delta_nablas.first[i];
                  nabla_b[i] += delta_nablas.second[i];
                }
        }

        // update weights and biases according to nabla_w's and nabla_b's and the learning rate
        for (size_t i = 0; i < m_nLayers-1; i++) {
                m_weights[i]   =  (1.0-learningRate*(lambda/(double)dataSetSize))*m_weights[i] - (learningRate/mini_batch_size)*nabla_w[i];
                m_biases[i]   -=  (learningRate/(double)mini_batch_size) * nabla_b[i];
        }

}

int NeuralNet::eval( Data validationData){

        // Return the number of test inputs for which the neural
        // network outputs the correct result

        int num_correct = 0;

        for(size_t i = 0; i < validationData.nEntries; i++) { // go through all validationData entries
                VectorXd out = feed_forward(validationData.getImgEntry(i));

                // find index of maximum element
                int predictedDigit = 0;
                for(int i = 0; i < out.size(); i++) {
                        if(out[i] > out[predictedDigit]) predictedDigit = i;
                }

                int actualDigit = validationData.getDigitEntry(i);
                if (actualDigit == predictedDigit) num_correct += 1;
        }

        return num_correct;

}



std::pair<  std::vector<MatrixXd>, std::vector<VectorXd>  > NeuralNet::backprop(const VectorXd img,const int digit){

        // Return a std::pair (nabla_b, nabla_w) representing the
        // gradient for the cost function C_x.  nabla_b and
        // nabla_w are layer-by-layer vectors of Eigen Vectors, similar
        // to m_biases and m_weights.

        std::vector<MatrixXd> nabla_w;
        std::vector<VectorXd> nabla_b;

        VectorXd activation = img;
        std::vector<VectorXd>     activations;

        activations.push_back(activation);

        VectorXd                    z;
        std::vector<VectorXd>       zs;

        // feedforward
        for(size_t i = 0; i <= 1; i++) {
                z = m_weights[i] * activation;
                zs.push_back(z);
                activation = sig(z);
                activations.push_back(activation);
        }

        nabla_w.resize(m_nLayers-1);
        nabla_b.resize(m_nLayers-1);

        VectorXd delta = cost_derivative_cross_entropy(activations[m_nLayers-1], digit_vector(digit), zs[m_nLayers-2]);
        nabla_b[m_nLayers-2] = delta;

        MatrixXd ttmp = activations[m_nLayers-2] * delta.transpose();
        nabla_w[m_nLayers-2] = ttmp.transpose();

        MatrixXd tmp_mat;
        VectorXd sp,tmp_vec;

        int index;
        for (size_t i = m_nLayers-2; i > 0; i--){ // go backwards through neural network layers.

          index = m_nLayers-2-i;

          sp = sig_prime(zs[index]);
          tmp_vec = m_weights[index+1].transpose() * delta;
          delta = tmp_vec.array() * sp.array();

          nabla_b[index] = delta;

          tmp_mat = activations[index] * delta.transpose();
          nabla_w[index] = tmp_mat.transpose();
        }

        return make_pair(nabla_w,nabla_b);

}




NeuralNet::~NeuralNet(){   // destructor

}
