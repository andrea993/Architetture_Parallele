#include <cuda.h>
#include <stdio.h>

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
	int i=threadIdx.y;
	int j=threadIdx.x;
	for (int k=0; k<N; k++)
	{
		*(C+ i*N+j)+=*(A+i*N + k) * *(B+k*N+j);
	}	
}

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


int main()
{
	int N=10;
	int NN=N*N;
	int *A, *B, *C_host, *C_device;
	int *ptrA, *ptrB;

	cudaMallocManaged(&A, NN*sizeof(int));
	cudaMallocManaged(&B, NN*sizeof(int));
	cudaMallocManaged(&C_device, NN*sizeof(int));
	cudaMallocManaged(&C_host, NN*sizeof(int));
	cudaDeviceSynchronize(); //attende che la memoria sia allocata

	for (int i=0, ptrA=A, ptrB=B ; i<N; i++)
	{
		for (int j=0; j<N; j++)
		{
			*ptrA=i+j;
			*ptrB=0;
			if (i==j)
				*ptrB=1;

			ptrA++;
			ptrB++;
		}
	}

	fprintf(stderr,"A=\n");
	printMtx(A, N);
	fprintf(stderr,"\n\nB=\n");
	printMtx(B, N);
	fprintf(stderr,"\n\n");

	host_mmul(A, B, C_host, N);

	dim3 blockPerGrid(1,1);
	dim3 threadPerBlock(10,10);

	global_mmul <<< blockPerGrid, threadPerBlock >>> (A,B,C_device,N);

	cudaDeviceSynchronize();

	fprintf(stderr,"C_host=\n");
	printMtx(C_host, N);
	fprintf(stderr,"\n\nC_device=\n");
	printMtx(C_device, N);
	fprintf(stderr,"\n");
	// devo usare la memoria shared, ipotizzo che N sia una potenza di 2, prima sceglire come usare i blocchi poi implementare <- è un ASSIGNMENT

	return 0;
}

