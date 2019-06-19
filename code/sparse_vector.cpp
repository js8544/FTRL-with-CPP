#include "sparse_vector.h"

sparse_vector::sparse_vector(std::vector< std::pair< long, double> >& content):vc(),iter(vc.begin()){
	for(long i=0;i<content.size();i++){
		vc.insert(content[i]);
	}
}



sparse_vector sparse_vector::operator+(double s){
	sparse_vector res(*this);

	sp_iter it = res.vc.begin();
	while(it!=res.vc.end()){
		it->second += s;
		it++;
	}

	return res;
}

sparse_vector sparse_vector::operator+(sparse_vector& p){
	sparse_vector res(*this);
	sp_iter it = p.vc.begin();
	
	while(it!=p.vc.end()){
		long i = it->first;
		double v = it->second;

		res.set_value(i,res.get_value(i)+v);

		it++;
	}

	return res;

}

sparse_vector& sparse_vector::operator+=(sparse_vector& p){
	sp_iter it = p.vc.begin();
	
	while(it!=p.vc.end()){
		long i = it->first;
		double v = it->second;

		set_value(i,get_value(i)+v);

		it++;
	}

	return *this;
}

sparse_vector& sparse_vector::operator-=(sparse_vector& p){
	sp_iter it = p.vc.begin();
	
	while(it!=p.vc.end()){
		long i = it->first;
		double v = it->second;

		set_value(i,get_value(i)-v);

		it++;
	}

	return *this;
}

sparse_vector sparse_vector::operator-(double s){
	sparse_vector res(*this);

	sp_iter it = res.vc.begin();
	while(it!=res.vc.end()){
		it->second -= s;
		it++;
	}

	return res;

}

sparse_vector sparse_vector::operator-(sparse_vector& p){
	sparse_vector res(*this);
	sp_iter it = p.vc.begin();
	
	while(it!=p.vc.end()){
		long i = it->first;
		double v = it->second;

		res.set_value(i,res.get_value(i)-v);

		it++;
	}

	return res;
}

sparse_vector sparse_vector::operator*(double s){
	sparse_vector res(*this);

	sp_iter it = res.vc.begin();
	while(it!=res.vc.end()){
		it->second *= s;
		it++;
	}

	return res;
}

sparse_vector sparse_vector::operator*(sparse_vector& p){
	sparse_vector res(*this);
	sp_iter it = p.vc.begin();
	
	while(it!=p.vc.end()){
		long i = it->first;
		double v = it->second;

		res.set_value(i,res.get_value(i)*v);

		it++;
	}

	return res;
}

double sparse_vector::dot(sparse_vector& p){
	double res = 0;
	sp_iter it = p.vc.begin();

	while(it!=p.vc.end()){
		long i = it->first;
		double v = it->second;

		res += v*get_value(i);

		it++; 
	}

	return res;
}

void sparse_vector::print_value(){
	sp_iter it = vc.begin();
	while(it!=vc.end()){
		cout<<it->first<<": "<<it->second<<endl;
		it++;
	}
}
