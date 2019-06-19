#include <iostream>
#include <complex>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <string>

extern "C"
{
#include "bmp.h"
#include <pthread.h>
#include <sys/time.h>
}

using namespace std;

const double mx=-2.5;
const double Mx=1;
const double my=-1;
const double My=1;

const int maxiter=1024;

struct ThreadData
{
	ThreadData(int idx0, int idx1, BITMAP &mtx): 
		idx0(idx0), idx1(idx1), mtx(mtx) {}

	const int idx0;
	const int idx1;
	BITMAP &mtx;
};

inline double nowSec()
{
	struct timeval t;
	struct timezone tzp;
	gettimeofday(&t, &tzp);
	return t.tv_sec + t.tv_usec*1e-6;
}

inline complex<double> pixel2c(int row, int col, int width, int height)
{
	return complex<double> (
			col/static_cast<double>(width-1)*(Mx-mx) + mx, 
			(height-row-1)/static_cast<double>(height-1)*(My-my) + my);
}

void* parLoop(void *arg)
{
	ThreadData* data=static_cast<ThreadData*>(arg);
	complex<double> c, z;

	for (int i=data->idx0; i<data->idx1; i++)
		for (unsigned j=0; j<data->mtx.width; j++)
		{
			c=pixel2c(i,j,data->mtx.width,data->mtx.height);
			int itr=0;
			z=0;
			while (z.real()*z.real()+z.imag()*z.imag()<2*2 && itr<maxiter)
			{
				z=z*z+c;
				itr++;		
			}
			COLORTRIPLE color = {0,0,0};
			if (itr < maxiter)
				color.green = color.blue = itr*255/maxiter;
			
			PIXEL(data->mtx,i,j)=color;
		}

	delete data;
	pthread_exit(NULL);
	return (void*)0;
}


int main(int argc, char **argv)
{

	if (argc != 3)
	{
		cout<<"Usage:"<<endl
			<<"mandelbrot [IMAGE_WIDTH] [N_THREADS]"<<endl;
		return -1;
	}

	int width=stoi(argv[1]);
	int height=width*(My-my)/(Mx-mx);

	int N=stoi(argv[2]);

	BITMAP mtx=CreateEmptyBitmap(height,width);

	vector<pthread_t> threads(N);

	double t_begin = nowSec();
	for(int i=0; i<N; i++)
	{
		int i0=i*height/N;
		int i1=(i+1)*height/N;

		ThreadData *d_ptr=new ThreadData(i0,i1,mtx);

		pthread_create(&threads[i],NULL,parLoop,static_cast<void*>(d_ptr));
	}

	for_each(threads.begin(), threads.end(),
			[](pthread_t x){pthread_join(x,NULL);});

	double t_end=nowSec();
	cout << "Elapsed time: " << t_end-t_begin  << "sec" << endl;	

	FILE* fp=fopen("out.bmp","wb");
	WriteBitmap(mtx,fp);
	fclose(fp);
	ReleaseBitmapData(&mtx);

	return 0;  
}

