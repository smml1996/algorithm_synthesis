from copy import deepcopy
from qstates import QuantumState, get_inner_product
import numpy as np
from numpy import linalg as LA

def get_helstrom_operator(p1: float, p2: float, qs1: QuantumState, qs2: QuantumState) -> np.matrix:
    '''We only use this for 1 qubit systems.
    p1 and p2 are probabilities of being in states qs1 and qs2 respectively.
    '''
    rho1 = np.matrix(qs1.get_density_matrix())
    rho2 = np.matrix(qs2.get_density_matrix())
    return p1*rho1 - p2*rho2

def get_eigen(rho: np.matrix):
    assert isinstance(rho, np.matrix)
    assert rho.shape[0] == 2
    assert rho.shape[1] == 2
    
    eigenvalues, eigenvectors = LA.eig(rho)
    return eigenvalues, eigenvectors

def get_optimal_rotation(prob1: float, prob2: float, qs1: QuantumState, qs2: QuantumState):
    hm_op = get_helstrom_operator(prob1, prob2, qs1, qs2)
    eigenvalues, eigenvectors = get_eigen(hm_op)
    assert len(eigenvalues) == 2
    assert len(eigenvectors) == 2
    
    assert np.inner(eigenvectors[0], eigenvectors[0]) == 1.0
    assert np.inner(eigenvectors[1], eigenvectors[1]) == 1.0
    a = eigenvectors[0, 0]
    b = eigenvectors[1, 0]
    c = eigenvectors[0, 1]
    d = eigenvectors[1, 1]
    U = np.matrix([[a, b], [c, d]])
    
    identity = np.eye(U.shape[0])
    U_dagger = np.conjugate(U.T)
    assert np.allclose(U_dagger @ U, identity, atol=1e-10)
    return U

def get_success_prob(p1: float, p2: float, qs1: QuantumState, qs2: QuantumState) -> float:
    return 1/2 + 1/2*np.sqrt(1-4*p1*p2*get_inner_product(qs1, qs2)**2)