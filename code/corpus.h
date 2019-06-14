#ifndef CORPUS_H_
#define CORPUS_H_

#include <fstream>
#include <iostream>
#include <vector>
#include <string>

#include "sparse_vector.h"

typedef sparse_vector SpVec;

using namespace std;

struct data_str{//data struct
	SpVec x;
	double y;
	data_str(SpVec X, double Y):x(X),y(Y){};
};

class corpus
{
private:
	ifstream *file;

public:
	long d;
	vector<data_str> data;
	corpus(ifstream *File);//when the file is given in a sparse pattern, i.e. feature:value
	long size();
	data_str &operator[](long i);
};

#endif