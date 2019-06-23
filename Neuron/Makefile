CXXFLAGS = -std=c++11 -Wall -O2
neuron: HebbianLearning.o
	g++ -o neuron HebbianLearning.o
neuron.o: HebbianLearning.cpp
	g++ $(CXXFLAGS) -c HebbianLearning.cpp
clean:
	rm -f *.o \
rm -f neuron

