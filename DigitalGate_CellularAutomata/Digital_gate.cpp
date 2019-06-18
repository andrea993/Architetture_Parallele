// Digital_gate.cpp: definisce il punto di ingresso dell'applicazione.



#include "automata.hpp"

#include <iostream>
#include <vector>

std::vector<std::vector<Automata::State>> or_gate(bool a, bool b)
{
	Automata::State _ = Automata::State::empty;
	Automata::State O = Automata::State::conductor;
	Automata::State H = Automata::State::electron_head;
	Automata::State T = Automata::State::electron_tail;

	std::vector<std::vector<Automata::State>> x = {
		{_, _, _, _, _, O, O, _, _, _, _, _, _},
		{O, O, O, O, O, _, _, O, _, _, _, _, _},
		{_, _, _, _, _, _, O, O, O, O, O, O, O},
		{O, O, O, O, O, _, _, O, _, _, _, _, _},
		{_, _, _, _, _, O, O, _, _, _, _, _, _}
	};

	if (a)
	{
		x[1][0] = T;
		x[1][1] = H;
	}

	if (b)
	{
		x[3][0] = T;
		x[3][1] = H;
	}

	return x;
}


int main()
{
	Automata automata(or_gate(1, 1));

	for (int i = 0; i < 12; i++)
	{
		std::cout << automata << std::endl;
		automata.Step();
	}
	
	system("pause");
	return 0;
}
