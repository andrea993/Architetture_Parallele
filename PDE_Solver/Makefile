CXXFLAGS = -std=c++11 -Wall -O2

pdeSolver: pdeSolver.o
	g++ -o pdeSolver pdeSolver.o
bmp.o: bmp.cu
	nvcc -c bmp.cu
pdeSolver.o: pdeSolver.cpp
	g++ $(CXXFLAGS) -c pdeSolver.cpp
clean:
	rm -f *.o \
rm -f pdeSolver

