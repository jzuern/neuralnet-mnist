#ifndef UTIL_H
#define UTIL_H

#include <Eigen/Dense>
#include "math.h"
#include <iostream>
#include "stdio.h"
#include <vector>
#include <cassert>
#include <utility>


using namespace Eigen;


inline double sig(const double z){

        // The sigmoid function for scalars

        return 1.0/(1.0+exp(-z));
}


inline VectorXd sig(const VectorXd z){
  
        // The sig function for a VectorXd class

        VectorXd result(z.size());

        for (int i = 0; i < z.size(); i++) {
                result[i] = (1.0/(1.0+exp(-z[i])));
        }

        return result;
}

inline double sig_prime(const double z){

        // Derivative of the sigmoid function

        return sig(z)*(1.0-sig(z));
}

inline VectorXd sig_prime(const VectorXd z){

        // Derivative of sigmoid function for VectorXd class

        VectorXd result(z.size());

        for (int i = 0; i < z.size(); i++) {
                result[i] = sig(z[i]) * (1.0 - sig(z[i]));
        }
        return result;
}


inline VectorXd digit_vector(const int digit){

        VectorXd vec(10); // vector of correct result

        for (int i = 0; i < 10; i++) {
                if(i == digit) vec[i] = 1.0;
                else {
                        vec[i] = 0.0;
                }
        }
        return vec;
}


#endif
