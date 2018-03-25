#include <iostream>
#include <complex>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <string>
#include <ctime>

extern "C"
{
#include "bmp.h"
#include <pthread.h>
}

using namespace std;

const double mx=-2.5;
const double Mx=1;
const double my=-1;
const double My=1;

const int maxiter=1024;

struct ThreadData
{
	ThreadData(int idx0, int idx1, BITMAP &mtx, const vector<COLORTRIPLE> &colors): 
		idx0(idx0), idx1(idx1), mtx(mtx), colors(colors) {}

	const int idx0;
	const int idx1;
	BITMAP &mtx;
	const vector<COLORTRIPLE> &colors;
};

inline COLORTRIPLE newColor(byte r, byte g, byte b)
{
	COLORTRIPLE color = {b, g, r};
	return color;
}

inline complex<double> pixel2c(int row, int col, int width, int height)
{
	return complex<double> (
			col/static_cast<double>(width)*(Mx-mx) + mx, 
			row/static_cast<double>(height)*(My-my) + my);
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
			while (abs(z)<2 && itr<maxiter-1)
			{
				z=z*z+c;
				itr++;		
			}
			PIXEL(data->mtx,i,j)=data->colors[itr];
		}

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

	const int maxiter=1024;

	vector<COLORTRIPLE> colors(maxiter);
	BITMAP mtx=CreateEmptyBitmap(height,width);

	for_each (colors.begin(), colors.end(), 
			[](COLORTRIPLE& x){x=newColor(rand()%256,rand()%256,rand()%256);});

	clock_t t_begin=clock();

	for(int i=0; i<N; i++)
	{
		int i0=i*height/N;
		int i1=(i+1)*height/N;    
		ThreadData d(i0,i1,mtx,colors);

		pthread_t thread_i;
		pthread_create(&thread_i,NULL,parLoop,static_cast<void*>(&d));
		pthread_join(thread_i, NULL);
	}

	clock_t t_end=clock();
	cout << "Elapsed time: " << double(t_end-t_begin)/CLOCKS_PER_SEC << endl;	

	FILE* fp=fopen("out.bmp","wb");
	WriteBitmap(mtx,fp);
	fclose(fp);
	ReleaseBitmapData(&mtx);

	return 0;  
}

