import os
from cmath import cos, isclose
from collections import deque
from math import ceil, floor, sin
from typing import *
from enum import Enum
import numpy as np
from sympy import Symbol, expand_trig, simplify, symbols, trigsimp

CONFIG_KEYS = ["name", "experiment_id", "min_horizon", "max_horizon", "output_dir", "algorithms_file", "hardware"]

def myfloor(val, d):
    m = 10**d
    if isinstance(val, complex):
        v = val*m
        answer = complex(floor(v.real)/m, floor(v.imag)/m)
        if isclose(answer.imag, 0.0, abs_tol=Precision.isclose_abstol):
            answer = answer.real
        return answer
    return floor(val*m)/m

def myceil(val, d):
    m = 10**d
    if isinstance(val, complex):
        v = val*m
        answer = complex(ceil(v.real)/m, ceil(v.imag)/m)
        if isclose(answer.imag, 0.0, abs_tol=Precision.isclose_abstol):
            answer = answer.real
        return answer
    return ceil(val*m)/m

def find_enum_object(val: str, Obj: Enum):
    for element in Obj:
        if val == element.value:
            return element
    return None

class Queue:
    queue: deque

    def __init__(self) -> None:
        self.queue = deque()

    def pop(self) -> Any:
        if self.is_empty():
            raise Exception("trying to pop empty queue")
        return self.queue.popleft()

    def push(self, v):
        self.queue.append(v)

    def is_empty(self) -> bool:
        return len(self.queue) == 0

    def len(self) -> int:
        return len(self.queue)

class Precision:
    PRECISION = 8  # round number to `PRECISION` floating point digits
    isclose_abstol = None
    rel_tol = None
    is_lowerbound = True
    @staticmethod
    def update_threshold():
        Precision.isclose_abstol = 1/(10**(Precision.PRECISION-1))  
        Precision.rel_tol = 1/(10**(Precision.PRECISION-1))  
        

def are_matrices_equal(arr1, arr2):
    if len(arr1) != len(arr2):
        return False
    
    for (row1, row2) in zip(arr1, arr2):
        if len(row1) != len(row2):
            return False
        
        for (v1, v2) in zip(row1, row2):
            if not isclose(v1, v2, abs_tol=Precision.isclose_abstol):
                return False
    return True

def is_matrix_in_list(matrix, matrix_list):
    for m in matrix_list:
        if are_matrices_equal(matrix, m):
            return True
    return False

class Pauli(Enum):
    I = 0
    X = 1
    Y = 2
    Z = 3 

def invert_dict(d: Dict[Any, Any]) -> Dict[Any, Any]:
    d_inverse = dict()
    for (key, value) in d.items():
        assert value not in d_inverse.keys()
        d_inverse[value] = key
    return d_inverse



def get_kraus_matrix_probability(matrix: List[List[float]], a0: complex, a1: complex, return_new_ampl=False):
    """

    Args:
        matrix (List[List[float]]): kraus matrix
        [
            [a, b]
            [c, d]
        ]
        a0 (complex): amplitude of |0>
        a1 (complex): amplitude of |1>
    """

    assert len(matrix) == 2

    for l in matrix:
        assert len(l) == 2
    a = matrix[0][0]
    b = matrix[0][1]
    c = matrix[1][0]
    d = matrix[1][1]
    new_a0 = a*a0 + b*a1
    new_a1 = c*a0 + d*a1
    prob = new_a0*np.conjugate(new_a0) + new_a1*np.conjugate(new_a1)
    assert prob <= 1.0
    if return_new_ampl:
        return prob, new_a0, new_a1
    return prob

def get_index(value, values):
    
    for (index, v) in enumerate(values):
        if v == value:
            return index
    raise Exception("no value in values")

def np_get_ground_state(eigenvalues, eigenvectors):
    """

    Args:
        eigenvalues (_type_): _description_
        eigenvectors (_type_): _description_

    Returns:
        _type_: return a pair (ground_energy, [eigenvectors_with_ground_energy])
    """    
    r_eigenvalue = min(eigenvalues)
    result = []
    for (eigenvalue, eigenvector) in zip(eigenvalues, eigenvectors):
        if isclose(r_eigenvalue, eigenvalue, rel_tol=Precision.rel_tol):
            result.append(eigenvector)
    return r_eigenvalue, result

class Belief:
    def __init__(self) -> None:
        self.probabilities = dict()
        
    def set_probability(self, key: Any, value: Any):
        self.probabilities[key] = value
        
    def add_probability(self, key: Any, value: Any):
        if key not in self.probabilities.keys():
            self.set_probability(key, value)
        else:
            self.probabilities[key] += value
        
    def get_probability(self, key: Any):
        if key in self.probabilities.keys():
            return self.probabilities[key]
        return 0.0
    
def my_simplify(expr: Any) -> Any:
    previous = expr
    after = trigsimp(expr) 
    print(f"[trigsimp] {previous} --> {after}")
    return after

def expand_trig_func(expr: Any) -> Any:
    '''apply the sum or double angle identities
    '''
    previous = expr
    after = expand_trig(expr) 
    print(f"[expand trig] {previous} --> {after}")
    return after

class SymbolBank:
    counter = 0
    def get_symbol(self):
        new_symbol = symbols(f'x{SymbolBank.counter}')
        SymbolBank.counter += 1
        return new_symbol
    
def get_cos_sin_exprs(cost_function) -> List[Any]:
    ''' returns a list of expressions that we need to replace.
    An expression that we need to replace is an expression to which a cosine or sine function is applied
    '''
    if len(cost_function.args) > 0:
        result = []
        for arg in cost_function.args:
            if isinstance(arg, cos) or isinstance(arg, sin):
                result.extend(arg.args[0])
            else:
                result.extend(get_cos_sin_exprs(arg))
        return result
    else:
        return []
    
def replace_cos_sin_exprs(cost_function) -> Any:
    '''replaces expression sorrounded by cos/sin expression such that each depends only of 1 variable
    '''
    exprs_to_replace = get_cos_sin_exprs(cost_function)
    exprs_to_symbols = dict()
    for expr in exprs_to_replace:
        new_symbol = SymbolBank.get_symbol()
        cost_function = cost_function.subs(expr, new_symbol)
        exprs_to_symbols[expr] = new_symbol
        
    return cost_function
        
        
    
    
    

            