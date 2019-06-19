#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "bmp.h"
#include <sys/time.h>

typedef struct _mandelbrotPars
{
	float mx;
	float Mx;
	float my;
	float My;
	int width;
	int height;
	BITMAP M;
	unsigned maxiter;
}mandelbrotPars;

double nowSec()
{
	struct timeval t;
	struct timezone tzp;
	gettimeofday(&t, &tzp);
	return t.tv_sec + t.tv_usec*1e-6;
}

__global__ void global_mandelbrot(mandelbrotPars pars)
{
	int row = pars.height - blockIdx.y -1;
	int col = blockIdx.x;

	float c_re = blockIdx.x/(pars.width-1.0)*(pars.Mx-pars.mx)+pars.mx;
	float c_im = blockIdx.y/(pars.height-1.0)*(pars.My-pars.my)+pars.my;

	int itr = 0;
	float z_re = 0, z_im = 0, z_re_tmp;

	while (z_re*z_re + z_im*z_im < 2*2 && itr<pars.maxiter)
	{
		z_re_tmp = z_re*z_re - z_im*z_im + c_re;
		z_im = 2*z_re*z_im + c_im;
		z_re = z_re_tmp;

		itr++;
	}
	
	COLORTRIPLE color = {0,0,0};
	if (itr < pars.maxiter)
		color.green = color.blue = itr*255/pars.maxiter;
		
	PIXEL(pars.M, row, col) = color;

}

int main(int argc, char **argv)
{

	if (argc != 2)
	{
		fprintf(stderr,"Usage:\nmandelbrot [IMAGE_WIDTH]\n");
		return -1;
	}

	mandelbrotPars pars;
	pars.mx = -2.5;
	pars.Mx = 1;
	pars.my = -1;
	pars.My = 1;
	pars.maxiter = 1024;
	
	pars.width=atoi(argv[1]);
	pars.height=pars.width*(pars.My-pars.my)/(pars.Mx-pars.mx);

	pars.M = CreateEmptyBitmap(pars.height, pars.width);
	cudaDeviceSynchronize();
	
	dim3 blockPerGrid(pars.width, pars.height, 1);
	dim3 threadPerBlock(1, 1, 1);

	double t_begin = nowSec();
	global_mandelbrot <<< blockPerGrid, threadPerBlock >>> (pars);
	cudaDeviceSynchronize();
	double t_end = nowSec();
	
	printf("Elapsed Time: %f sec\n", t_end - t_begin);

	FILE* fp = fopen("out.bmp","wb");
	WriteBitmap(pars.M, fp);
	fclose(fp);
	ReleaseBitmapData(&pars.M);;

	return EXIT_SUCCESS;
}
