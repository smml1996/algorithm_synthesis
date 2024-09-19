from cmath import isclose
from typing import Any, List, Optional
from sympy import *
from qpu_utils import Precision, get_complex, int_to_bin, bin_to_int

class QuantumState:
    sparse_vector : Dict # map basis states to sympy-Symbolic
    substitutions: List = []

    def __init__(self, init_basis: Optional[int] = 0, 
                 init_amplitude = complex(1.0, 0.0), dimension=2):
        self.sparse_vector = dict()
        self.insert_amplitude(init_basis, init_amplitude)
        self.dimension = dimension

    def __str__(self) -> str:
        result = ""
        for (key, value) in self.sparse_vector.items():
            if not isinstance(value, complex):
                value = complex(value, 0.0)
            if len(result) > 0:
                result += " + "
            result += f"{value}|{int_to_bin(key)}>"
        return result
    
    def __repr__(self):
        return str(self)

    def get_amplitude(self, basis: int) -> Any:
        if basis in self.sparse_vector:
            return self.sparse_vector.get(basis)
        else:
            return 0
        
    def is_qubit(self) -> bool:
        if len(self.sparse_vector) > 2:
            return False
        
        for basis in self.sparse_vector.keys():
            if basis > 1:
                return False
        return True
    
    def is_qubit_0(self) -> bool:
        assert self.is_qubit()
        return isclose(simplify(self.get_amplitude(0)), 1)
    
    def get_qubit_amplitudes(self) -> Tuple:
        return self.get_amplitude(0), self.get_amplitude(1)
    
    def insert_amplitude(self, basis: int, amplitude: Any) -> bool:
        amplitude = get_complex(simplify(amplitude))
        if isclose(amplitude.real, 0.0, abs_tol=Precision.isclose_abstol) and isclose(amplitude.imag, 0.0, abs_tol=Precision.isclose_abstol):
                return False

        self.sparse_vector[basis] = simplify(amplitude)
        return True

    def add_amplitude(self, basis: int, amplitude: Any) ->  bool:
        amplitude = get_complex(simplify(amplitude))

        
        if isclose(amplitude.real, 0.0, abs_tol=Precision.isclose_abstol) and isclose(amplitude.imag, 0.0, abs_tol=Precision.isclose_abstol):
            # if both the real and the imaginary part of the amplitude are 0 then we return False because we do not add any amplitude
            return False
        
        prev_amplitude = self.get_amplitude(basis)
        current_amplitude = get_complex(simplify(prev_amplitude + amplitude))
        if isclose(current_amplitude.real, 0.0, abs_tol=Precision.isclose_abstol) and isclose(current_amplitude.imag, 0.0, abs_tol=Precision.isclose_abstol):
            # if the new amplitude is 0 in both the real and imaginary part we delete this key
            del self.sparse_vector[basis]
            return False
        self.sparse_vector[basis] = current_amplitude
        return True

    def normalize(self):
        sum_ = 0
        for val in self.sparse_vector.values():
            sum_ += simplify(val*conjugate(val))

        norm = simplify(sqrt(sum_))

        for key in self.sparse_vector.keys():
            val = simplify(self.sparse_vector[key] / norm)
            self.sparse_vector[key] = val

    def __eq__(self, other):
        assert isinstance(other, QuantumState)
        if len(self.sparse_vector.keys()) != len(other.sparse_vector.keys()):
            return False
        
        factor = None
        for (key, val1) in self.sparse_vector.items():
            if key not in other.sparse_vector.keys():
                return False
            val2 = other.sparse_vector[key]
            if factor is None:
                factor = val1/val2
            current_factor = val1/val2
            if current_factor != factor:
                return False
        return True
    
    def get_density_matrix(self) -> List[List[int]]:
        result = []
        for row in self.dimension:
            temp = []
            for col in self.dimension:
                temp.append(self.get_amplitude(row) * self.get_amplitude(col).conj())
            result.append(temp)
        return result
    
    def single_partial_trace(self, rho=None, index=0):
        if rho is None:
            rho = self.get_density_matrix()

        initial_dim = len(rho)
        assert (initial_dim % 2) == 0

        # initialize the result as a matrix full of zeros
        result = []
        for _ in range(initial_dim/2):
            temp = 0
            for _ in range(initial_dim/2):
                temp.append(0)                
            result.append(temp)

        for ket in range(initial_dim):
            bin_ket = int_to_bin(ket, zero_padding=initial_dim)
            bin_new_ket = bin_ket[:index] + bin_ket[index+1:]
            index_new_ket = bin_to_int(bin_new_ket)
            for bra in range(initial_dim):
                bin_bra = int_to_bin(bra, zero_padding=initial_dim)
                bin_new_bra = bin_bra[:index] + bin_bra[index+1:]
                index_new_bra = bin_to_int(bin_new_bra)
                assert result[index_new_ket][index_new_bra] == 0
                if bin_ket[index] == bin_bra[index]:
                    result[index_new_ket][index_new_bra] = rho[ket][bra]
        assert len(result) == initial_dim/2
        return result

    def multi_partial_trace(self, rho=None, remove_indices: List[int]=[0]) -> List[List[float]]:
        if rho is None:
            rho = self.get_density_matrix()
        for index in remove_indices:
            result = self.single_partial_trace(rho, index)
        return result
    
    def get_trace(self, rho: List[List[float]]):
        result = 0
        for i in range(len(rho)):
            result += rho[i][i]
        return result


def are_states_lists_equal(qsl1: List[QuantumState], qsl2: List[QuantumState]) -> bool:    
    if len(qsl1) != len(qsl2):
        return False
    
    for qs1 in qsl1:
        if not (qs1 in qsl2):
            return False
    return True

def is_list_subset(subset: List[QuantumState], all_set: List[QuantumState]) -> bool:
    if len(all_set) < len(subset):
        return False
    
    for qs in subset:
        if not (qs in all_set):
            return False
    return True

def are_qs_lists_equal(qsl1: List[QuantumState], qsl2: List[QuantumState]) -> bool:    
    if len(qsl1) != len(qsl2):
        return False
    
    for qs1 in qsl1:
        assert isinstance(qs1, QuantumState)
        if not (qs1 in qsl2):
            return False
    return True

def does_observable_exists(obs_to_qs: Dict, quantum_states: List) -> Optional[int]:
    ''' obs_to_qs: Dictonary that maps an int --> List of quantum states
    '''
    for (obs, value) in obs_to_qs.items():
        if are_qs_lists_equal(value, quantum_states):
            return obs
    return None

def get_fidelity(qstate1: QuantumState, qstate2: QuantumState) -> float:
    inner_product = 0
    for (key, val1) in qstate1.sparse_vector.items():
        val2 = qstate2.get_amplitude(key)
        inner_product += val1*val2
    return inner_product

