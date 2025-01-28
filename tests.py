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
def test_sat(name: str):
    sat = SATFormula(f"sat/{name}.txt")
    all_assignments = sat.find_all_true_assignments()

    H = sat.get_hamiltonian()
    energies = get_energies(H, sat.n_variables)

    low_energy_bitstrings = get_lowest_energies_bitstrings(energies, sat.n_variables)
    assert len(low_energy_bitstrings) == len(all_assignments)

    for b in low_energy_bitstrings:
        assignment = dict()
        for (index, x) in enumerate(b):
            if x == "1":
                assignment[index+1] = True
            else:
                assert x == "0"
                assignment[index+1] = False
        assert sat.evaluate_solution(assignment)
        assert assignment in all_assignments

if __name__ == "__main__":
    test_graph("g0")