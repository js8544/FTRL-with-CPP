#include "LR.h"
#include "corpus.h"
#include "sparse_vector.h"
#include <iostream>
#include <cmath>
#include <string>
#include <iomanip>
#include <omp.h>
// #include <omp.h>
// #include <pthread.h>
using namespace std;

inline int sgn(double a){
	return (0<a)-(a>0);
}

class FTRL
{
private:
	//Number of Features
	int d;

	//Parameters
	double alpha;
	double beta;
	double l1;
	double l2;

	//Decision Function
	LR lr;

	//Model
	SpVec w;
	SpVec z;
	SpVec n;

public:
	FTRL():alpha(0.5),beta(1.0),l1(1.0),l2(1.0),d(4){};

	FTRL(int D, double Alpha, double Beta, double L1, double L2):d(D),alpha(Alpha),beta(Beta),l1(L1),l2(L2){};

	double perdict(SpVec& x){
		return lr.decision(w,x);
	}

	double update(SpVec& x, double y){
		#pragma omp parallel for
		for(int i = 0;i < d;++i){
			if(abs(z.get_value(i)) > l1){
				#pragma omp critical
				w.set_value(i,-1.0/((beta+sqrt(n.get_value(i)))/alpha+l2)*(z.get_value(i)-sgn(z.get_value(i))*l1));
			}
			else{
				#pragma omp critical
				w.set_value(i,0);
			}
		}
		double p = perdict(x);
		SpVec g = lr.gradient(p,y,x);

		#pragma omp parallel for
		for(int i = 0;i < d;i++){
			double temp_g = (p-y)*(x.get_value(i));
			g.set_value(i, temp_g);
			double sigma = (1.0/alpha)*(sqrt(n.get_value(i)+(temp_g*temp_g))-sqrt(n.get_value(i)));
			double temp_z = z.get_value(i)+temp_g-sigma*w.get_value(i);
			double temp_n = n.get_value(i)+temp_g*temp_g;
			
			#pragma omp critical
			z.set_value(i,temp_z);
			#pragma omp critical
			n.set_value(i,temp_n);
		}

		return lr.loss(p,y);
	}

	void train(corpus& data){
		cout<<"training "<<data.size()<<" data"<<endl;
		double correct = 0;
		double wrong = 0;

		for(int i=0;i<data.size();i++){
			cout<<"data "<<i<<" ";
			int p = (perdict(data[i].x)>0.5);
			double loss = update(data[i].x,data[i].y);

			if(p==data[i].y){
				correct++;
			}
			else{
				wrong++;
			}
			cout<<"loss: "<< loss <<" ";
			cout<<"accuracy: "<<correct/(correct+wrong)<<endl;



		}
		cout<<"trained weight:"<<endl;
		w.print_value();
	}

	double test(corpus& data){
		cout<<"testing "<<data.size()<<" data"<<endl;
		double correct = 0;
		double wrong = 0;

		#pragma omp parallel for
		for(int i=0;i<data.size();i++){

			cout<<"data "<<i<<":"<<perdict(data[i].x)<<"--"<<data[i].y<<endl;
			int p=0;
			if(perdict(data[i].x)>0.5){
				p = 1;
			}
			if(p==data[i].y){
				correct++;
			}
			else{
				wrong++;
			}
		}
		return correct/(correct+wrong);
	}

	void save(ofstream* FILE){
		sp_iter it = w.vc.begin();
		while(it!=w.vc.end()){
			(*FILE)<<it->first<<": "<<it->second<<endl;
			it++;
		}
	}

	void load(ifstream* FILE){
		string line;
		while(getline(*FILE,line)){
			vector<string> temp = parse_feature(line,":");
			
			long x = stoi(temp[0]);
			double v = stod(temp[1]);
			w.set_value(x,v);
		}
	}
};

int main(int argc, char const *argv[])
{


	if(string(argv[1])=="-train"){
		if(argc!=4){
			cout<<"Usage: ./ftrl -train [train data] [model save location]\n";
			return 0;
		}
		cout<<"creating train set\n";
		ifstream FILE;
		FILE.open(argv[2]);
		corpus train_set(&FILE);

		FTRL ftrl(train_set.d, 0.5, 1, 1, 1);

		cout<<"training\n";
		ftrl.train(train_set);


		ofstream SAVE;
		SAVE.open(argv[3]);
		cout<<"saving trained model\n";
		ftrl.save(&SAVE);

		FILE.close();
		SAVE.close();
	}

	else if(string(argv[1])=="-test"){
		if(argc!=4){
			cout<<"Usage: ./ftrl -test [test data] [model save location]\n";
			return 0;
		}
		
		cout<<"creating test set\n";
		ifstream TEST;
		TEST.open(argv[2]);
		corpus test_set(&TEST);

		FTRL ftrl(test_set.d, 0.5, 1, 0, 0);

		cout<<"loading model\n";
		ifstream LOAD;
		LOAD.open(argv[3]);
		ftrl.load(&LOAD);


		cout<<"testing\n";
		cout<<"accuracy: "<<ftrl.test(test_set)<<endl;

		LOAD.close();
		TEST.close();
	}

	else{
		cout<<"Usage: ./ftrl -train [train data] [model save location]\nUsage: ./ftrl -test [model save location] [test data]\n";
	}

	
	
	return 0;
}
