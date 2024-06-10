from utils import *


# functions to test max-cut problem

def test_graph(name: str):
    g = UndirectedGraph(f"graphs/{name}.txt")
    solution, val = g.get_max_cut_solutions()

    H = g.get_hamiltonian()
    energies = get_energies(H, g.n_vertices)

    low_energy_bitstrings = get_lowest_energies_bitstrings(energies, g.n_vertices)
    assert len(low_energy_bitstrings) == len(solution)
    for b in low_energy_bitstrings:
        b_cut_value = g.get_bitstring_cut_value(b)
        assert b_cut_value == val
        assert b in solution

# functions to test SAT



if __name__ == "__main__":
    test_graph("g0")