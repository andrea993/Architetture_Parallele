#include <vector>
#include <iostream>
#include "neuron.hpp"

float fitness(float u1, float u2)
{
	std::cout << int(int(u1) && int(u2)) << std::endl;
	return int(u1) && int(u2);
}

int main()
{
	Neuron n(2, Neuron::Activation::heaviside);

	bool trained = false;

	float nu=0.5;

	std::vector<std::vector<float>> dataset_u = { {0,0}, {0,1}, {1,0}, {1,1} };
	std::vector<float> dataset_y(dataset_u.size());

	for (int i=0; i<dataset_u.size(); i++)
		dataset_y[i]=fitness(dataset_u[i][0], dataset_u[i][1]);

	while (!trained)
	{
		trained = true;
		for (int i=0; i<n.Ninp(); i++)
		{
			if (int(n.Perf(dataset_u[i])) != int(dataset_y[i]))
			{
				trained=false;
				break;
			}
		}
		if (trained)
			break;

		for (int i=0; i<n.Ninp(); i++)
		{
			float dw_0 = nu*dataset_u[i][0]*dataset_y[i];
			float dw_1 = nu*dataset_u[i][1]*dataset_y[i];
			
			std::cout<<"dw0:" << dw_0 << " dw1:" <<dw_1 << std::endl;

			n.setWeight(0, n.Weigth(0)+dw_0);
			n.setWeight(1, n.Weigth(1)+dw_1);
		}
	}

	std::cout << "trained" << std::endl;


}
