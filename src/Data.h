#ifndef DATA_H
#define DATA_H

#include "string.h"
#include "util.h"
#include <Eigen/Dense> // eigen library for matrix vector stuff
#include <sstream>
#include <fstream>
#include <vector>
#include <iostream>

using namespace Eigen;

class Data {

  // Structure of Data Class Instantiation:
  // MNIST raw data: 70,000 Data sets
  // structure of each data entry:
  // (imgData,correctDigit) tuple, where:
  //    imgData: 28x28 (28x28=784) vector of 8bit grey values of MNIST images
  //    correctDigit: 10-dimensional binary vector of correct digit


public:
        void load_images_from_file(const std::string filename);
        Data();                                     // empty constructor
        size_t nEntries;                            // number of entries in data set
        int getDigitEntry(int index);     // digit getter function
        VectorXd getImgEntry(int index);  // image getter function

private:

        std::vector<VectorXd> imgData; // vector of VectorXds containing the list of pixels of each image
        std::vector<int> correctDigit; // vector of correct digits

};

#endif
