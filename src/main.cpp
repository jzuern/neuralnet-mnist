// include standard libraries header
#include <fstream>
#include <iostream>
#include "stdio.h"

// include header for the 2 classes
#include "NeuralNet.h"
#include "Data.h"


int main (int argc, char *argv[]) {

        if (argc != 3) {
          std::cerr << "Please provide the correct number of command-line arguments.\n";
          std::cerr << "Possible command is: ./nnet data/mnist_train.csv data/mnist_test.csv 30\n";

          return 1;
        }


        int nHidden; // number of hidden neurons
        int nIn  = 28*28; //number of input neurons (equal to the number of pixels of input image)
        int nOut = 10; // number of output neurons  (equal to the number of possible classes: 0,1,2,...,9)

        std::cout << "Please name the integer number of neurons in the hidden layer. A value of 30-60 is recommended\n";
        std::cin >> nHidden;

        if (std::cin.fail()) {
            std::cerr << "Not a valid integer\n";
            return 1;
        }

        std::cout << "NeuralNet program started\n";


        // instantiate NeuralNet object
        NeuralNet nnet(nIn, nHidden, nOut);

        // instantiate Data objects
        Data trainingData, validationData;

        std::string trainData_file = argv[1];
        std::string validationData_file = argv[2];


        std::ifstream train_file_stream (trainData_file );
        std::ifstream validation_file_stream ( validationData_file );

        // Check to see if file opening succeeded
        if ( !train_file_stream ) {
              std::cout<<"error: Training csv file could not be found\n";
              return 1;
        }
        if ( !validation_file_stream ) {
              std::cout<<"error: Training csv file could not be found\n";
              return 1;
        }


        std::cout << "Allocating training data...\n";
        trainingData.load_images_from_file(trainData_file);          // load data file into memory

        std::cout << "Allocating validation data...\n";
        validationData.load_images_from_file(validationData_file);  // load data file into memory

        std::cout << "Allocation successfully completed\n";

        // training parameter:
        const double learning_rate    = 0.1; // learning rate
        const size_t num_epochs       = 30;  // number of epochs
        const size_t mini_batch_size  = 10;  // size of mini batch
        const double lambda_l2        = 5.0; // L2 regularization parameter

        // train NeuralNet with Stochastic Gradient Descent Method
        std::cout << "Starting Training....\n";
        nnet.train(trainingData, num_epochs, mini_batch_size, learning_rate, lambda_l2, validationData);

        // program finished
        return 0;
}
