#ifndef SPARSE_VECTOR_H_
#define SPARSE_VECTOR_H_

#include <unordered_map>
#include <vector>
#include <iostream>
typedef std::unordered_map<long, double> sp_type;
typedef std::unordered_map<long, double>::iterator sp_iter;

using namespace std;
class sparse_vector
{
private:
	sp_iter iter;

public:
	sp_type vc;
	
	sparse_vector(vector<pair<long,double> >& content);
	
	sparse_vector(void):vc(),iter(vc.begin()){};
	
	sparse_vector(const sparse_vector& sp):vc(sp.vc),iter(vc.begin()){};

	double get_value(long i);

	void set_value(long i, double v);

	inline long size(){
		return vc.size();
	};

	sparse_vector operator+(double s);
	sparse_vector operator+(sparse_vector& p);
	
	sparse_vector& operator+=(sparse_vector& p); 
	sparse_vector& operator-=(sparse_vector& p); 

	sparse_vector operator-(double s);
	sparse_vector operator-(sparse_vector& p);

	sparse_vector operator*(double s);
	sparse_vector operator*(sparse_vector& p);	

	double dot(sparse_vector& p);

	void print_value();

};

#endif
