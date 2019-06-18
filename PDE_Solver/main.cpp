#include <vector>
#include <iostream>
#include <string>
#include <iomanip>
#include <sys/time.h>

const float TOLERANCE=1e-3;
const int MAXITER=1024;

inline double nowSec()
{
	struct timeval t;
	struct timezone tzp;
	gettimeofday(&t, &tzp);
	return t.tv_sec + t.tv_usec*1e-6;
}

std::ostream& operator<<(std::ostream &os, std::vector<std::vector<double>> &mtx)
{

	for (int i=0; i<mtx.size(); i++)
	{
		for (int j=0; j<mtx[i].size(); j++)
		{
			os<<std::fixed<<std::setprecision(4)<<mtx[i][j]<<"\t";
		}
		os<<std::endl;
	}

	return os;
}


int main(int argc, char **argv)
{

	if (argc != 2)
	{
		std::cout<<"Usage:"<<std::endl<<
			"PDAsolver [matrix size]"<<std::endl;

		return -1;
	}



	int N=std::stoi(argv[1]);


	std::vector<std::vector<double>> mtx(N+2, std::vector<double>(N+2,0));


	for(int i=0; i<mtx.size();i++)
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
		for (int i=1; i<N; i++)
			for (int j=1; j<N; j++)
			{
				old=mtx[i][j];
				mtx[i][j]=0.2*(mtx[i][j]+mtx[i-1][j]+mtx[i+1][j]+mtx[i][j-1]+mtx[i][j+1]);

				sumdiff+=abs(mtx[i][j]-old);
			}
		itr++;
	}
	while(sumdiff/tot>=TOLERANCE && MAXITER>itr);
	double t1=nowSec();

	std::cout<<"Elapsed time: "<<(t1-t0)<<"sec"<<std::endl;

#ifdef OUTP
	std::cout<<mtx<<std::endl;
#endif

	return 0;

}
