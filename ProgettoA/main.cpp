#include <iostream>
#include <complex>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <string>

extern "C"
{
#include "bmp.h"
#include <sys/time.h>
}

using namespace std;

const double mx=-2.5;
const double Mx=1;
const double my=-1;
const double My=1;

const int maxiter=1024;

struct RandomColor : COLORTRIPLE
{
	RandomColor()
	{
			blue=static_cast<byte>(rand()%256);
			green=static_cast<byte>(rand()%256);
			red=static_cast<byte>(rand()%256);
	}
};

inline double nowSec()
{
	struct timeval t;
	struct timezone tzp;
	gettimeofday(&t, &tzp);
	return t.tv_sec + t.tv_usec*1e-6;
}

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

int main(int argc, char **argv)
{

	if (argc != 2)
	{
		cout<<"Usage:"<<endl
			<<"mandelbrot [IMAGE_WIDTH]"<<endl;
		return -1;
	}

	int width=stoi(argv[1]);
	int height=width*(My-my)/(Mx-mx);


	vector<RandomColor> colors(maxiter);
	BITMAP mtx=CreateEmptyBitmap(height,width);

	complex<double> c, z;
	double t_begin = nowSec();
	for (int i=0; i<height; i++)
		for (int j=0; j<width; j++)
		{
			c=pixel2c(i,j,width,height);
			int itr=0;
			z=0;
			while (abs(z)<2 && itr<maxiter-1)
			{
				z=z*z+c;
				itr++;		
			}
			PIXEL(mtx,i,j)=colors[itr];
		}

	double t_end=nowSec();
	cout << "Elapsed time: " << t_end-t_begin  << "sec" << endl;

	FILE* fp=fopen("out.bmp","wb");
	WriteBitmap(mtx,fp);
	fclose(fp);
	ReleaseBitmapData(&mtx);

	return 0;  
}

