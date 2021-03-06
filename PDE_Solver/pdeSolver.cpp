#include <vector>
#include <iostream>
#include <string>
#include <iomanip>
#include <cmath>

extern "C" 
{
	#include <sys/time.h>
}

const float TOLERANCE=1e-5;
const int MAXITER=1024;

inline double nowSec()
{
	struct timeval t;
	struct timezone tzp;
	gettimeofday(&t, &tzp);
	return t.tv_sec + t.tv_usec*1e-6;
}

std::ostream& operator<<(std::ostream &os, std::vector< std::vector<float> > &mtx)
{

	for (unsigned i=0; i<mtx.size(); i++)
	{
		for (unsigned j=0; j<mtx[i].size(); j++)
			os<<std::fixed<<std::setprecision(4)<<mtx[i][j]<<"\t";
		
		os<<std::endl;
	}

	return os;
}

int main(int argc, char **argv)
{
	if (argc != 2)
	{
		std::cout<<"Usage:"<<std::endl<<
			"pdeSolver [matrix size]"<<std::endl;

		return -1;
	}

	int N=std::stoi(argv[1]);
	std::vector<std::vector<float>> mtx(N+2, std::vector<float>(N+2,0));

	for(unsigned i=0; i<mtx.size();i++)
	{
		mtx[i][0] = 1;
		mtx[0][i] = 1;
		mtx[N+1][i] = 1;
		mtx[i][N+1] = 1;
	}

	int itr=0;
	float sumdiff, old, tot=N*N;

	double t0=nowSec();
	do
	{
		sumdiff = 0;
		for (int i=1; i<N+1; i++)
			for (int j=1; j<N+1; j++)
			{
				old=mtx[i][j];
				mtx[i][j]=0.25*(mtx[i-1][j]+mtx[i+1][j]+mtx[i][j-1]+mtx[i][j+1]);

				sumdiff+=std::fabs(mtx[i][j]-old);
			}
		itr++;
	}
	while(sumdiff/tot>=TOLERANCE && MAXITER>itr);
	
	double t1=nowSec();
	
#ifdef PRINT	
	std::cout << mtx << std::endl;
#endif


	std::cout<<"Elapsed time: "<<(t1-t0)<<"sec"<<std::endl
		<< "Iterations: " << itr << std::endl;

	return 0;

}
