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
		heaviside,
		linear,
		sigmoid,
		tanh
	};

	Neuron(unsigned Ninp=0, Activation a=Activation::heaviside) : w(Ninp,1),a(a),w0(0) {}

	void setWeight(int i, float w_i) { w[i] = w_i; }
	float Weigth(int i) const { return w[i]; }
	float W0() const { return w0; }
	void setW0(float w) { w0=w; };
	unsigned Ninp() const { return w.size(); }

	float Perf(const std::vector<float> &u)
	{
		if (u.size() != w.size())
			throw std::logic_error("wrong size input vector in Perf()");

		float y=0;
		for (int i=0; i<w.size(); i++)
			y+=w[i]*u[i];

		return y*Perfactv(y);
	}


private:
	
	float Perfactv(float u)
	{
		switch(a)
		{
			case Activation::heaviside:
				return (u>w0) ? 1 : 0;
			case Activation::linear:
				return u*w0;
			case Activation::sigmoid:
				return 1.0/(1.0 + exp(-u/w0));
			case Activation::tanh:
				return (exp(u/w0)-exp(-u/w0))/(exp(u/w0)+exp(-u/w0));

		}
	}
	
	std::vector<float> w;
	float w0;
	Activation a;
	
};


#endif
