### Implementation of a basic neural network for the recognition of the MNIST data set of handwritten digits


The neural network has been designed in order to process the MNIST data given as a raw 8bit pixel list from a csv file.


The reinforcement learning technique is implemented using the backpropagation algorithm with a stochastic gradient descent algorithm. The training data set is hereby divided into mini-batches, which are selected in a random fashion.

This project was inspired by the [great online-book Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html) by Michael Nielsen. I reimplemented the therein developed python implementations in C++. It has one hidden layer with an arbitrary number of hidden neurons that can be specified by the user. Additionally, I implemented the L2 regularization approach which reduces the risks of overfitting to the training data set, as well as the cross-entropy cost function.



## Build requirements

- [eigen library](http://eigen.tuxfamily.org/index.php?title=Main_Page) must be installed

## Build

compile with provided makefile:

```
$ make
```

## Usage

Run with
```
$ ./nnet path/to/training_file.csv path/to/validation_file.csv number_of_hidden_neurons
```
where the 'training_file.csv' contains training images and the 'validation_file.csv' contains validation images in order to detect the accuracy of the neural network. The number of neurons in the hidden layer is given as the last argument (A value of 30-60 is recommended)

The [MNIST csv data set](http://pjreddie.com/projects/mnist-in-csv/) must be in a local directory.

## Results

After 20 epochs, the neural network with 50 hidden neurons achieved an accuracy of ~90% on the validation data set, which is remarkable given its simple architecture.
