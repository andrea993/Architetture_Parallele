Matrix_mult: Matrix_mult.o
	nvcc -o Matrix_mult Matrix_mult.o 
Matrix_mult.o: Matrix_mult.cu
	nvcc -c Matrix_mult.cu 
clean:
	rm -f *.o \
rm -f Matrix_mult

