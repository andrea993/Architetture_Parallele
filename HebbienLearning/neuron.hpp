#ifndef NEURON_HPP
#define NEURON_HPP

#include <vector>
#include <cmath>
#include <stdexcept>

class Neuron
{
public:
	enum class Activation
	{
		signum,
		linear,
		sigmoid,
		tanh
	};

	Neuron(unsigned Ninp=0, Activation a=Activation::sigmoid) : w(Ninp,0),w0(-3),a(a) {}

	void setWeight(int i, float w_i) { w[i] = w_i; }
	float Weight(int i) const { return w[i]; }
	float W0() const { return w0; }
	void setW0(float w) { w0=w; };
	unsigned Ninp() const { return w.size(); }

	float Perf(const std::vector<float> &u) const
	{
		if (u.size() != w.size())
			throw std::logic_error("wrong size input vector in Perf()");

		float y=0;
		for (int i=0; i<int(w.size()); i++)
			y+=w[i]*u[i];

		return Perfactv(y);
	}


private:
	
	float Perfactv(float u) const
	{
		switch(a)
		{
			case Activation::signum:
				return (u>w0) ? 1 : -1;
			case Activation::linear:
				return u*w0;
			case Activation::sigmoid:
				return 1.0/(1.0 + exp(-u/w0));
			case Activation::tanh:
				return (exp(u/w0)-exp(-u/w0))/(exp(u/w0)+exp(-u/w0));

		}

		return 0;
	}
	
	std::vector<float> w;
	float w0;
	Activation a;
	
};


#endif
