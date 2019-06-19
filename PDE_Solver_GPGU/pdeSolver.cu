#include <vector>
#include <iostream>
#include <string>
#include <iomanip>
#include <sys/time.h>
#include <cuda.h>
#include <cstdio>
#include <cmath>

const int MAXITER = 1024;
const int DIVISOR = 512;

enum Color { red, black };

#define AT(mtx, width, row, column)  \
        mtx[(row) * (width) + (column)]

inline double nowSec()
{
	struct timeval t;
	struct timezone tzp;
	gettimeofday(&t, &tzp);
	return t.tv_sec + t.tv_usec*1e-6;
}

void printMtx(float *mtx, int size)
{
	for (unsigned i=0; i<size; i++)
	{
		for (unsigned j=0; j<size; j++)
			std::cout<<std::fixed<<std::setprecision(4)<<AT(mtx,size,i,j)<<"\t";
		
		std::cout<<std::endl;
	}
} 

__global__ void global_cellSolve(float *mtx, int dim, int M, float* itr)
{
	float tot = (M-2)*(M-2);
	Color c = red;
	
	int i = (M-2)-1 - (blockIdx.y * blockDim.y + threadIdx.y) + 1; //si somma 1 perchè ci sono N*N thread 	
	int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
	
	do
	{
		
		if ((i % 2 == 0 && c == red) ||
			(i % 2 == 1 && c == black))
		{
			AT(mtx,M,i,j) = 0.2*(AT(mtx,M,i-1,j)+AT(mtx,M,i+1,j)+AT(mtx,M,i,j-1)+AT(mtx,M,i,j+1));	
		}
		
		c = c == red ? black : red;
		atomicAdd(itr, 0.5); //aver computato un solo colore corrisponde a mezza iterazione
		
		__syncthreads();
	}
	while(MAXITER > *itr/tot);
}

int main(int argc, char **argv)
{
	float* mtx;
	float* itr;
	
	if (argc != 2)
	{
		std::cout<<"Usage:"<<std::endl<<
			"pdeSolver [matrix size]"<<std::endl;

		return -1;
	}

	int N = std::atoi(argv[1]);
	int M = N + 2;
	
	if (N % DIVISOR)
	{
		std::cerr << "N deve essere un multiplo di " << DIVISOR <<std::cout;
		return -1;
	}
	
	int dim = N/DIVISOR;
	
	cudaMallocManaged(&mtx, M*M*sizeof(float));
	cudaMallocManaged(&itr, sizeof(float));
	cudaDeviceSynchronize();
	
	for(unsigned i=0; i<M; i++)
	{
		AT(mtx,M,i,0) = 1;
		AT(mtx,M,0,i) = 1;
		AT(mtx,M,M-1,i) = 1;
		AT(mtx,M,i,M-1) = 1;
	}
	
	dim3 blockPerGrid(dim, dim, 1);
	dim3 threadPerBlock(DIVISOR, DIVISOR, 1); 

	double t_begin = nowSec();
	global_cellSolve <<< blockPerGrid, threadPerBlock >>> (mtx, dim, M, itr);
	cudaDeviceSynchronize();
	double t_end = nowSec();

#ifdef PRINT	
	printMtx(mtx, M);
#endif

	std::cout<<"Elapsed time: "<<(t_end-t_begin)<<"sec"<<std::endl;
	
	cudaFree (mtx);
	cudaFree (itr);

	return 0;

}
