#include "automata.hpp"

// friend
std::ostream& operator<<(std::ostream &os, Automata& auotomata)
{
	for (int i=0; i < auotomata.Rows(); i++)
		for (int j = 0; j < auotomata.Cols(); j++)
		{
			auto &x = auotomata.grid[i][j];
			switch (x)
			{
			case Automata::State::electron_head:
				os << '+';
				break;
			case Automata::State::electron_tail:
				os << '-';
				break;
			case Automata::State::conductor:
				os << '=';
				break;
			case Automata::State::empty:
				os << ' ';
				break;
			default:
				continue;
			}

			if (j == auotomata.Cols() - 1)
				os << std::endl;
		}

	return os;

}

// public
void Automata::Step()
{
	/*
			*** RULES ***
	Empty -> Empty
	Electron head -> Electron tail
	Electron tail -> Conductor
	Conductor -> Electron head if exactly one or two of the neighbouring cells are electron heads, or remains Conductor otherwise.
	*/

	auto gridnew = grid;
	unsigned count;

	for (int i = 0; i < Rows(); i++)
		for (int j = 0; j < Cols(); j++)
		{
			auto &x = grid[i][j];
			auto &y = gridnew[i][j];

			switch (x)
			{
			case State::electron_head:
				y = State::electron_tail;
				break;
			case State::electron_tail:
				y = State::conductor;
				break;
			case State::conductor:
				count = Neighbors(i, j);
				if (count == 1 || count == 2)
					y = State::electron_head;
				break;
			default:
				continue;
			}
		}
	grid = gridnew;
}

// private
unsigned Automata::Neighbors(int i, int j)
{
	unsigned count = 0;
	for (int n = i - 1; n <= i + 1; n++)
		for (int k = j - 1; k <= j + 1; k++)
		{
			if (n == i && k == j ||
				n < 0 || n >= Rows() ||
				k < 0 || k >= Cols())
				continue;

			if (grid[n][k] == Automata::State::electron_head)
				count++;
		}

	return count;
}