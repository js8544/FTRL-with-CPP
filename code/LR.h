#ifndef LR_H_
#define LR_H_

#include <math.h>
#include <time.h>
#include <string>
#include <iostream>
#include <vector>
#include "sparse_vector.h"

typedef sparse_vector SpVec;

class LR
{
public:
	LR(){};
	inline double decision(SpVec w, SpVec x){
		//Sigmoid Decision Function
		return 1.0 / (1.0 + exp(w.dot(x)));
	}

	inline double loss(double p, double y){
		//Log Loss Function
		return -y*log(p)-(1-y)*log(1-p);
	}
	inline SpVec gradient(double p, double y, SpVec x){
		//Gradient Function
		return x*(p-y);
	}
};

#endif