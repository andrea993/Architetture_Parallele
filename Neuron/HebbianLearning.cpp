#include <vector>
#include <iostream>
#include <ostream>
#include "neuron.hpp"

const int NINP = 2;
const int NSET = 4;
const float nu = 0.01;


bool trained(const std::vector<std::vector<float>> &data, const Neuron &n)
{
	bool train = true;
	for (int i=0; i<NSET; i++)
	{
		std::vector<float> u(data[i].begin(),data[i].begin()+NINP);
		if (n.Perf(u) != data[i].back())
		{
			train = false;
			break;
		}
	}
	return train;
}

int main(int argc, char **argv)
{

	if(argc != 2)
	{
USAGE:
		std::cerr<<"Usage"<<std::endl<<
			"neuron [OR\\AND]"<<std::endl;

		return -1;
	}

	Neuron n(NINP, Neuron::Activation::signum);

	unsigned itr=0;

	std::vector<std::vector<float>> data_or = {
		{-1,	-1,	-1},
		{-1,	 1,	 1},
		{ 1,	-1,	 1},
		{ 1,	 1,	 1}}; 

	std::vector<std::vector<float>> data_and = {
		{-1,	-1,	-1},
		{-1,	 1,	-1},
		{ 1,	-1,	-1},
		{ 1,	 1,	 1}}; 

	std::vector<std::vector<float>> data;

	if ("OR" == std::string(argv[1]))
	{
		data = data_or;
		n.setW0(-3);
	}
	else if ("AND" == std::string(argv[1]))
	{
		data = data_and;
		n.setW0(0);
	}
	else
		goto USAGE;


	int row = 0;
	while (!trained(data, n))
	{
		for (int i=0; i<NINP; i++)
		{
			float dw = nu*data[row][i]*data[row].back();
			n.setWeight(i, n.Weight(i) + dw);
		}
		row = (row + 1) % NSET;	
		itr++;
	}


	std::cout << "trained in " << itr << " iterations" << std::endl;
	
	std::cout << n.W0() << '\t';
	for (int i=0; i< NINP; i++)
		std::cout << n.Weight(i) << '\t';
	
	std::cout << std::endl;

	return 0;
}
