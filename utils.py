from typing import *
from enum import Enum
import numpy as np
from sympy import I

def split_line(line: str, expected_elements: int = 1, split_by=" ") -> List[str]:
    """split line by some criteria (split_by)
    Args:
        line (str): line to split
        expected_elements (int): number of elements to expect after splitting
        split_by (char): splitting criteria
    Returns:
        List[str]: list with each element
    """       
    elements = line.split(split_by)
    assert len(elements) == expected_elements
    return elements

class UndirectedGraph:
    def __init__(self, path: str) -> None:
        """init function for an undirected graph, receives as input a file that contains the specification of the graph

        Args:
            path (str): file where the graph is stored. The first line of the file consists of two integers: the number n of vertices and the number m of edges. The following m lines consists on two integers 1 <= a,b <= n.
        """ 
        self.edges = dict()
        self.n_vertices = -1
        self.n_edges = -1

        f = open(path)
        lines = f.readlines()
        elements = split_line(lines[0], 2)
        self.n_vertices = int(elements[0])
        self.n_edges = int(elements[1])
        
        if len(lines) - 1 != self.n_edges:
            raise Exception("Number of edges specified on first line do not match") 
        
        for i in range(1, self.n_edges + 1):
            elements = split_line(lines[i], 2)
            source = int(elements[0])-1
            target = int(elements[1])-1
            
            assert 0 <= source < self.n_vertices
            assert 0 <= target < self.n_vertices
            if min(source, target) not in self.edges.keys():
                self.edges[min(source, target)] = []
            self.edges[min(source, target)].append(max(source, target))
        f.close()

class Pauli(Enum):
    I = 0
    X = 1
    Y = 2
    Z = 3

def get_pauli_matrix(p: Pauli) -> np.array:
    if p == Pauli.I:
        return np.array([[1, 0], 
                         [0, 1]])
    elif p == Pauli.X:
        return np.array([[0, 1], 
                         [1, 0]])
    elif p == Pauli.Y:
        return np.array([[0, -I], 
                         [I, 0]])
    else:
        assert p == Pauli.Z
        return np.array([[1, 0], 
                         [0, -1]])

def get_matrix(position: int, p: Pauli, system_size:int) -> np.array:
    """computes a matrix that is the kronecker product of system_size pauli matrices. all of them are identity matrix except the one at 'position'.
    Args:
        position (int): position where the term p is
        p (Pauli): a pauli matrix
        system_size (int): total number of qubits

    Returns:
        np.array: I^{position}\otimes p \otimes I^{system_size-position-1}
    """    
    result = np.array([1])
    assert  0 <= position < system_size
    for i in range(system_size):
        if i == position:
            result = np.kron(result, get_pauli_matrix(p))
        else:
            result = np.kron(result, get_pauli_matrix(Pauli.I))
    return result
    
def get_max_cut_hamiltonian(graph: UndirectedGraph, decomposed: bool = True) -> np.array:
    """ Computes the hamiltonian whose ground state is the solution to the max cut problem of a given graph.
    Args:
        graph (UndirectedGraph): graph where we want to compute the max cut problem
        decomposed (bool, optional): if set to False returns a list of length 1 that represent the kronecker product of graph.n_vertices pauli matrices. Otherwise it return a list containing graph.n_edges. Defaults to True. [TODO: erase if never implemented]
    Returns:
        List[List[Pauli]]: The computed hamiltonian.
            Ex. Consider a Hamiltonian H that can be decomposed in H = H_1 + H_2. Furthermore H_1 can we written as the kronecker product of Pauli matrices. Thus if $H_1 = X\otimes I$ and $H_2 = Z\otimes I$ Then we return a list containing [[X, I], [Z, I]]. [TODO: erase if never implemented]
    """    
    
    # Recall. The purpose is to divide the graph into two sets. A vertex can either be in the set 0 or in the set 1.

    # We consider a quantum system with the same number of qubits as the number of vertices in the given graph. If a vertex i is in set 0 then its corresponding qubit q_i will have a spin 1 otherwise it will have a spin -1.
    num_qubits = graph.n_vertices 

    assert  0 < num_qubits  <= 5

    result = 0
    id_matrix = get_matrix(1, Pauli.I, num_qubits)
    for (source, targets) in graph.edges.items():
        for target in targets:
            # 1/2 (s1*s2 - 1)
            s1_matrix = get_matrix(source, Pauli.Z, num_qubits)
            s2_matrix = get_matrix(target, Pauli.Z, num_qubits)
            result = result + (s1_matrix*s2_matrix - id_matrix)/2

    return result

def compute_hamiltonian_energy(H: np.array, state: np.array) -> float:
    "the energy is given by <state|H|state>"
    bra_state = np.transpose(np.conjugate(state))
    return np.matmul(bra_state, np.matmul(H, state))[0][0]

