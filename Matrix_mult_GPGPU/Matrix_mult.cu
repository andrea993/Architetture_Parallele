#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>

inline double nowSec()
{
	struct timeval t;
	struct timezone tzp;
	gettimeofday(&t, &tzp);
	return t.tv_sec + t.tv_usec*1e-6;
}

__host__ void host_mmul (int *A, int *B, int *C, int N)
{
	for(int i=0; i<N; i++)
	{
		for (int j=0; j<N; j++)
		{
			for(int k=0; k<N; k++)
			{
				*(C+ i*N+j)+=*(A+i*N + k) * *(B+k*N+j);
			}
		}
	}
}

__global__ void global_mmul (int *A, int *B, int *C, int N)
{
	int i=blockIdx.y;
	int j=blockIdx.x;
	for (int k=0; k<N; k++)
	{
		*(C+ i*N+j)+=*(A+i*N + k) * *(B+k*N+j);
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
	int NN=N*N;
	int i;
	int *A, *B, *C_host, *C_device;
	int *ptrA, *ptrB;

	cudaMallocManaged(&A, NN*sizeof(int));
	cudaMallocManaged(&B, NN*sizeof(int));
	cudaMallocManaged(&C_device, NN*sizeof(int));
	cudaMallocManaged(&C_host, NN*sizeof(int));
	cudaDeviceSynchronize(); //attende che la memoria sia allocata

	for (i=0, ptrA=A, ptrB=B ; i<N; i++)
	{
		for (int j=0; j<N; j++)
		{
			*ptrA=i*N+j;
			*ptrB=0;
			if (i==j)
				*ptrB=1;

			ptrA++;
			ptrB++;
		}
	}

#ifdef PRINT
	fprintf(stderr,"A=\n");
	printMtx(A, N);
	fprintf(stderr,"\n\nB=\n");
	printMtx(B, N);
	fprintf(stderr,"\n\n");
#endif

	double t_begin_cpu = nowSec();
	host_mmul(A, B, C_host, N);
	double t_end_cpu = nowSec();

	dim3 blockPerGrid(N,N);
	dim3 threadPerBlock(1,1);

	double t_begin_gpu = nowSec();
	global_mmul <<< blockPerGrid, threadPerBlock >>> (A,B,C_device,N);
	cudaDeviceSynchronize();
	double t_end_gpu = nowSec();
	
#ifdef PRINT
	fprintf(stderr,"C_host=\n");
	printMtx(C_host, N);
	fprintf(stderr,"\n\nC_device=\n");
	printMtx(C_device, N);
	fprintf(stderr,"\n");
#endif

	printf("Elapsed time CPU: %f sec\n", t_end_cpu  - t_begin_cpu);
	printf("Elapsed time GPU: %f sec\n", t_end_gpu  - t_begin_gpu);
	
	cudaFree(A);
	cudaFree(B);
	cudaFree(C_device);
	cudaFree(C_host);

	return 0;
}

