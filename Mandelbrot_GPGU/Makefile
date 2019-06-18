mandelbrot: bmp.o mandelbrot.o
	nvcc -o mandelbrot bmp.o mandelbrot.o 
bmp.o: bmp.cu
	nvcc -c bmp.cu
mandelbrot.o: mandelbrot.cu
	nvcc -c mandelbrot.cu
clean:
	rm -f *.o \
rm -f mandelbrot

