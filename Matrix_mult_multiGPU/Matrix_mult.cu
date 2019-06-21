#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

const int PARTITION_SIZE = 2;

#define AT(mtx, width, row, column)  \
        mtx[(row) * (width) + (column)]
	

inline double nowSec()
{
	struct timeval t;
	struct timezone tzp;
	gettimeofday(&t, &tzp);
	return t.tv_sec + t.tv_usec*1e-6;
}


__global__ void global_mmul (int *A, int *B, int *C, int N, int Ndev, int dev)
{
	int NperDev = N/Ndev;
	int i = (NperDev)-1 - (blockIdx.y * blockDim.y + threadIdx.y) + NperDev*dev;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	int iAC = i % NperDev;
	
	int i_part  = i % PARTITION_SIZE;
	int j_part = j % PARTITION_SIZE;
	
	int rowPerPart = N/PARTITION_SIZE;
	
	__shared__ int Apart[PARTITION_SIZE][PARTITION_SIZE];
	__shared__ int Bpart[PARTITION_SIZE][PARTITION_SIZE];
	
	AT(C, N, iAC, j) = 0;
	for (int n = 0; n < rowPerPart; n++)
	{
		Apart[i_part][j_part] = AT(A, N, iAC, n*PARTITION_SIZE + j_part);
		Bpart[i_part][j_part] = AT(B, N, n*PARTITION_SIZE + i_part, j);
		__syncthreads();
	 
		for (int k=0; k<PARTITION_SIZE; k++)
			AT(C, N, iAC, j) +=  Apart[i_part][k]*Bpart[k][j_part];

	} 
}

#ifdef PRINT
void printMtx (int **m, int N, int width, int height)
{	
	for (int i=0; i<N; i++)
	{
		for (int j=0; j<N; j++)
		{	
			if (width > height)
				printf("%d\t", AT(m[i/height], width, i%height, j));
			else
				printf("%d\t", AT(m[j/width], width, i, j%width));
		}
		puts("\n");
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
	
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	
	if (N % PARTITION_SIZE)
	{
		printf ("error: N must be a multiple of %d\n", PARTITION_SIZE);
		return -1;
	}
	
	if (N % nDevices)
	{
		printf ("error: N must be a multiple of nDevices=%d\n", nDevices);
		return -1;
	}
	
	unsigned NN=N*N;
	unsigned NNperDevice = NN/nDevices;
	unsigned NperDevice = N/nDevices;
	
	int Nblocks = N/PARTITION_SIZE;
	int NblocksPerDevice =Nblocks/nDevices;

	int **A_d, **B_d, **C_d;
	int **A_h, *B_h, **C_h;
	A_d = (int**)malloc(nDevices * sizeof(int*));
	B_d = (int**)malloc(nDevices * sizeof(int*));
	C_d = (int**)malloc(nDevices * sizeof(int*));
	A_h = (int**)malloc(nDevices * sizeof(int*));
	C_h = (int**)malloc(nDevices * sizeof(int*));
	
	B_h = (int*)malloc(sizeof(int)*NN);
	for (int i=0; i<nDevices; i++)
	{
		A_h[i] = (int*)malloc(sizeof(int)*NNperDevice);
		C_h[i] = (int*)malloc(sizeof(int)*NNperDevice);
		cudaSetDevice(i);
		cudaMalloc(&A_d[i], sizeof(int)*NNperDevice);
		cudaMalloc(&C_d[i], sizeof(int)*NNperDevice);
		cudaMalloc(&B_d[i], sizeof(int)*NN);
	}
	cudaDeviceSynchronize();

	for (int i=0; i<N; i++)
	{
		for (int j=0; j<N; j++)
		{								
			AT(A_h[i/NperDevice], N, i%NperDevice, j) = ((i == j) ? 1 : 0);
			AT(B_h, N, i, j) = i*N+j;
		}
	}

	for (int i=0; i<nDevices; i++)
	{
		cudaSetDevice(i);
		cudaMemcpy(B_d[i], B_h, NN*sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(A_d[i], A_h[i], NNperDevice*sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(C_d[i], C_h[i], NNperDevice*sizeof(int),cudaMemcpyHostToDevice);
	}
	cudaDeviceSynchronize();
	
	dim3 blockPerGrid(Nblocks,NblocksPerDevice);
	dim3 threadPerBlock(PARTITION_SIZE,PARTITION_SIZE);
	
	double t_begin = nowSec();
	for (int i=0; i<nDevices; i++)
	{
		cudaSetDevice(i);		
		global_mmul <<< blockPerGrid, threadPerBlock >>> (A_d[i],B_d[i],C_d[i],N,nDevices,i);
	}
	cudaDeviceSynchronize();
	double t_end = nowSec();

	
	for (int i=0; i<nDevices; i++)
	{
		cudaSetDevice(i);
		cudaMemcpy(C_h[i], C_d[i], NNperDevice*sizeof(int),cudaMemcpyDeviceToHost);
	}
	cudaDeviceSynchronize();
	

#ifdef PRINT
	fprintf(stderr,"A=\n");
	printMtx(A_h, N, N, NperDevice);
	fprintf(stderr,"\n\nB=\n");
	printMtx(&B_h, N, N, N);
	fprintf(stderr,"\n\nC=\n");
	printMtx(C_h, N, N, NperDevice);
	fprintf(stderr,"\n");
#endif

	printf("Elapsed time: %f sec\n", t_end  - t_begin);

	for (int i=0; i<nDevices; i++)
	{
		cudaFree(A_d[i]);
		cudaFree(B_d[i]);
		cudaFree(C_d[i]);
	}
	
	free(A_h);
	free(B_h);
	free(B_d);
	
	return 0;
}

