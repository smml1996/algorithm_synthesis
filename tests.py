from utils import *

# GENERAL functions
def get_energies(H: List[List[float]], num_qubits: int):
    energies = dict()
    for i in range(2**num_qubits):
        l = [0 for _ in range (2**num_qubits)]
        l[i] = 1
        state = np.transpose(np.array([l]))
        energies[i] = compute_hamiltonian_energy(H, state)
    return energies

def get_lowest_energies_bitstrings(energies: Dict[int, float], num_qubits: int):
    answer = set()
    min_val = min(energies.values())
    for i in range(2**num_qubits):
        if energies[i] == min_val:
            b = "{0:b}".format(i)
            while len(b) < num_qubits:
                b = "0" + b
            assert len(b) == num_qubits
            assert b not in answer
            answer.add(b)
    return answer
            
# functions to test max-cut problem
def get_bitstring_cut_value(g: UndirectedGraph, bitstring: str):
    A = set()
    for (index, bit) in enumerate(bitstring):
        if bit == "1":
            A.add(index)

    # compute cut value
    cut_value = 0
    for (source, targets) in g.edges.items():
        for target in targets:
            if source in A:
                if target not in A:
                    cut_value +=1
            elif target in A:
                cut_value += 1
    return cut_value

def get_max_cut_solutions(g):
    solutions = set()
    max_cut_val = -1
    for i in range(2**g.n_vertices):
        b = "{0:b}".format(i)
        while len(b) < g.n_vertices:
            b = "0" + b
        val = get_bitstring_cut_value(g, b)
        if val > max_cut_val:
            max_cut_val = val
            solutions = set()
            solutions.add(b)
        elif val == max_cut_val:
            solutions.add(b)
    return solutions, max_cut_val


def test_graph(name: str):
    g = UndirectedGraph(f"graphs/{name}.txt")
    solution, val = get_max_cut_solutions(g)

    H = get_max_cut_hamiltonian(g)
    energies = get_energies(H, g.n_vertices)

    low_energy_bitstrings = get_lowest_energies_bitstrings(energies, g.n_vertices)
    assert len(low_energy_bitstrings) == len(solution)
    for b in low_energy_bitstrings:
        b_cut_value = get_bitstring_cut_value(g, b)
        assert b_cut_value == val
        assert b in solution

# functions to test SAT



if __name__ == "__main__":
    test_graph("g0")