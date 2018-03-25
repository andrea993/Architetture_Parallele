CFLAGS = -O2
CXXFLAGS = -std=c++11 -Wall -O2
mandelbrot: bmp.o main.o
	g++ -o mandelbrot bmp.o main.o -lpthread
bmp.o: bmp.c
	gcc $(CFLAGS) -c bmp.c
main.o: main.cpp
	g++ $(CXXFLAGS) -c main.cpp 
clean:
	rm -f *.o \
rm -f mandelbrot \
rm out.bmp


