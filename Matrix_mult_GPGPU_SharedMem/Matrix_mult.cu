#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

const int PARTITION_SIZE = 32;

#define AT(mtx, width, row, column)  \
        mtx[(row) * (width) + (column)]
	

inline double nowSec()
{
	struct timeval t;
	struct timezone tzp;
	gettimeofday(&t, &tzp);
	return t.tv_sec + t.tv_usec*1e-6;
}


__global__ void global_mmul (int *A, int *B, int *C, int N)
{
	int i = N-1 - (blockIdx.y * blockDim.y + threadIdx.y);
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	int i_part  = i % PARTITION_SIZE;
	int j_part = j % PARTITION_SIZE;
	
	int rowPerPart = N/PARTITION_SIZE;
	
	__shared__ int Apart[PARTITION_SIZE][PARTITION_SIZE];
	__shared__ int Bpart[PARTITION_SIZE][PARTITION_SIZE];
	
	AT(C, N, i, j) = 0;
	for (int n = 0; n < rowPerPart; n++)
	{
		Apart[i_part][j_part] = AT(A, N, i, n*PARTITION_SIZE + j_part);
		Bpart[i_part][j_part] = AT(B, N, n*PARTITION_SIZE + i_part, j);

		__syncthreads();
	 
		for (int k=0; k<PARTITION_SIZE; k++)
			AT(C, N, i, j) +=  Apart[i_part][k]*Bpart[k][j_part];
	}

}

#ifdef PRINT
void printMtx (int *m, int N)
{
	for (int i=0; i<N*N; i++)
	{
		if (i>0 && i%N == 0)
			fprintf(stderr, "\n");

		fprintf(stderr, "%d\t",*m);

		m++;
	}
}
#endif

int main(int argc, char **argv)
{
	if (argc != 2)
	{
		puts("Usage: Matrix_mult [N]\n");
		return -1;
	}
	
	int N=atoi(argv[1]);
	
	if (N % PARTITION_SIZE)
	{
		printf ("error: N must be a multiple of %d\n", PARTITION_SIZE);
		return -1;
	}
	
	unsigned NN=N*N;
	int Nblocks = N/PARTITION_SIZE;

	int *A, *B, *C;
	cudaMallocManaged(&A, NN*sizeof(int));
	cudaMallocManaged(&B, NN*sizeof(int));
	cudaMallocManaged(&C, NN*sizeof(int));
	cudaDeviceSynchronize();

	for (int i=0; i<N; i++)
	{
		for (int j=0; j<N; j++)
		{
			AT(A, N, i, j) = (i == j) ? 1 : 0;
			AT(B, N, i, j) = i*N + j;	
		}
	}

#ifdef PRINT
	fprintf(stderr,"A=\n");
	printMtx(A, N);
	fprintf(stderr,"\n\nB=\n");
	printMtx(B, N);
	fprintf(stderr,"\n\n");
#endif
	
	dim3 blockPerGrid(Nblocks,Nblocks);
	dim3 threadPerBlock(PARTITION_SIZE,PARTITION_SIZE);
	
	double t_begin = nowSec();
	global_mmul <<< blockPerGrid, threadPerBlock >>> (A,B,C,N);
	cudaDeviceSynchronize();
	double t_end = nowSec();

#ifdef PRINT
	fprintf(stderr,"\n\nC=\n");
	printMtx(C, N);
	fprintf(stderr,"\n");
#endif

	printf("Elapsed time: %f sec\n", t_end  - t_begin);

	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	
	return 0;
}

