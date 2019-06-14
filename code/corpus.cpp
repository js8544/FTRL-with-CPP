#include "corpus.h"
#include <algorithm>
using namespace std;

vector<string> parse_feature(string s, const string del){
	size_t pos = 0;
	string token;
	vector<string> ret;

	while((pos = s.find(del))!=string::npos){
		token = s.substr(0,pos);
		ret.push_back(token);
		s.erase(0,pos+del.size());
	}
	ret.push_back(s);
	return ret;
}

corpus::corpus(ifstream* File){
	file = File;
	d = 0;
	string line;
	while(getline(*file,line)){
		vector<string> f = parse_feature(line," ");

		int y = (f[0]=="1")?1:0;
		SpVec fv;
		for(int i = 1;i < f.size();i++){
			
			vector<string> temp = parse_feature(f[i],":");
			
			long x = stoi(temp[0]);
			double v = stod(temp[1]);
			fv.set_value(x,v);
			if(x>d) d = x;
		}
		data.push_back(data_str(fv,y));
	}
}

long corpus::size(){
	return data.size();
}

data_str &corpus::operator[](long i){
	return data[i];
}