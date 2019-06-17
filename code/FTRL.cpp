#include "LR.h"
#include "corpus.h"
#include "sparse_vector.h"
#include <iostream>
#include <cmath>
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
		for(int i = 0;i < d;++i){
			if(abs(z.get_value(i)) > l1){
				w.set_value(i,-1.0/((beta+sqrt(n.get_value(i)))/alpha+l2)*(z.get_value(i)-sgn(z.get_value(i))*l1));
			}
			else{
				w.set_value(i,0);
			}
		}
		double p = perdict(x);
		SpVec g = lr.gradient(p,y,x);
		for(int i = 0;i < d;i++){
			g.set_value(i, (p-y)*(x.get_value(i)));
			double sigma = (1.0/alpha)*(sqrt(n.get_value(i)+(g.get_value(i)*g.get_value(i)))-sqrt(n.get_value(i)));
			z.set_value(i,z.get_value(i)+g.get_value(i)-sigma*w.get_value(i));
			n.set_value(i,n.get_value(i)+g.get_value(i)*g.get_value(i));
		}

		return lr.loss(p,y);
	}

	void train(corpus& data){
		cout<<"training "<<data.size()<<" data"<<endl;
		for(int i=0;i<data.size();i++){
		// for(int i=0;i<2000;i++){
			cout<<"round "<<i<<endl;
			cout<<"loss: "<<update(data[i].x,data[i].y)<<endl;
		}
		cout<<"trained weight:"<<endl;
		w.print_value();
	}

	double test(corpus& data){
		cout<<"testing "<<data.size()<<" data"<<endl;
		double correct = 0;
		double wrong = 0;

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
};

int main(int argc, char const *argv[])
{
	cout<<"creating train set\n";
	ifstream FILE;
	FILE.open(argv[1]);
	corpus train_set(&FILE);

	cout<<"creating test set\n";
	ifstream TEST;
	TEST.open(argv[2]);
	corpus test_set(&TEST);

	cout<<"training\n";
	FTRL ftrl(train_set.d, 0.5, 1, 0, 0);
	ftrl.train(train_set);

	cout<<"testing\n";
	cout<<"accuracy: "<<ftrl.test(test_set)<<endl;
	return 0;
}