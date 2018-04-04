package mandelbrotMT;

import java.awt.Color;
import java.awt.image.BufferedImage;

public class ParLoop extends Thread
{

	public ParLoop(int idx0, int idx1, BufferedImage mtx, Color[] colors)
	{
		this.idx0=idx0;
		this.idx1=idx1;
		this.mtx=mtx;
		this.colors=colors;
	}

	@Override
	public void run()
	{
		for(int i=idx0; i<idx1; i++)
			for (int j=0; j<mtx.getWidth(); j++)
			{
				double c_re=pixel2c_re(i, j, mtx.getWidth(),mtx.getHeight());
				double c_im=pixel2c_im(i, j, mtx.getWidth(),mtx.getHeight());
				int itr=0;
				
				double z_re=0;
				double z_im=0;
				
				while(z_re*z_re + z_im*z_im < 2*2 && itr<Mandelbrot.maxiter-1)
				{
					double z_re2=z_re*z_re - z_im*z_im + c_re;
					z_im=2*z_re*z_im+c_im;
					z_re=z_re2;
					itr++;			
				}
				
				mtx.setRGB(j,i,colors[itr].getRGB());
				
			}
		
	}
	
	private double pixel2c_re(int row, int col, int width, int height)
	{
		return col/(double)(width)*(Mandelbrot.Mx-Mandelbrot.mx) + Mandelbrot.mx;
	}
	
	private double pixel2c_im(int row, int col, int width, int height)
	{
		return row/(double)(height)*(Mandelbrot.My-Mandelbrot.my) + Mandelbrot.my;
	}


	int idx0;
	int idx1;
	BufferedImage mtx;
	Color[] colors;
	

}

