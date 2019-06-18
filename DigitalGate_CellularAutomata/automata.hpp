#ifndef _AUTOMATA_HPP_
#define _AUTOMATA_HPP_

#include <vector>
#include <assert.h>
#include <ostream>


class Automata
{
	friend std::ostream& operator<<(std::ostream &os, Automata& auotomata);

public:
	enum class State
	{
		empty,
		electron_head,
		electron_tail,
		conductor
	};

	Automata(int rows, int cols) : grid(rows)
	{
		assert(rows > 0 && cols > 0);
		for (auto &row : grid)
			row.resize(cols,  State::empty);
	}

	Automata(std::vector<std::vector<State>> &grid) : grid(grid) {}

	size_t Rows() const { return grid.size(); }
	size_t Cols() const { return grid[0].size(); }

	std::vector<State>& operator[] (unsigned i) { return grid[i]; }

	void Step();

private:
	std::vector<std::vector<State>> grid;

	unsigned Neighbors(int i, int j);


};

#endif
