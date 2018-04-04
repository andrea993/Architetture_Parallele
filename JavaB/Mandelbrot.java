package mandelbrotMT;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import java.util.Random;

public class Mandelbrot
{
	
	static final double mx=-2.5;
	static final double Mx=1;
	static final double my=-1;
	static final double My=1;
	

	static final int maxiter=1024;

	public static void main(String[] args)
	{
		if(args.length != 2)
		{
			System.out.println("Usage:\n"
			+"mandelbrot [IMAGE_WIDTH] [N_THREADS]");
			System.exit(-1);
		}
		int width=Integer.parseInt(args[0]);
		int height=(int) (width*(My-my)/(Mx-mx));

		int N=Integer.parseInt(args[1]);
		
		Random rnd=new Random();
		Color[] colors=new Color[maxiter];
		for (int i=0; i<maxiter; i++)
			colors[i]=new Color(rnd.nextInt(255),rnd.nextInt(255),rnd.nextInt(255));
		
		BufferedImage mtx = new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB);
		
		
		long beginT = System.currentTimeMillis();

		ParLoop[] pl=new ParLoop[N];
		for(int i=0; i<N; i++)
		{
			int i0=i*height/N;
			int i1=(i+1)*height/N;
			
			pl[i]=new ParLoop(i0, i1, mtx, colors);
			pl[i].start();
		}
		
		for (int i=0; i<N; i++)
			try
			{
				pl[i].join();
			}
			catch (InterruptedException e)
			{
				e.printStackTrace();
			}
		
		long endT = System.currentTimeMillis();
		
		System.out.println("Elapsed time: " + (endT-beginT)*1E-3 + "sec");
		
		File outputfile = new File("out.bmp");
		try
		{
			ImageIO.write(mtx, "bmp", outputfile);
		}
		catch(IOException e)
		{
			System.out.println("Error on image write");
			e.printStackTrace();
		}
		
		
		
			

	}

}
