#ifndef CORPUS_H_
#define CORPUS_H_

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#include "sparse_vector.h"

typedef sparse_vector SpVec;

using namespace std;

vector<string> parse_feature(string s, const string del);

struct data_str{//data struct
	SpVec x;
	double y;
	data_str(SpVec X, double Y):x(X),y(Y){};
};

class corpus
{
public:
	ifstream *file;
	long d;
	vector<data_str> data;
	corpus(ifstream *File);//when the file is given in a sparse pattern, i.e. feature:value
	long size();
	data_str &operator[](long i);
};

#endif