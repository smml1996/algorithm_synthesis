from cmath import isclose, sqrt
from copy import deepcopy
from typing import Any, List, Optional, Dict, Tuple
import numpy as np
from numpy import conjugate
from qpu_utils import Precision, get_complex, int_to_bin, bin_to_int
from scipy.linalg import expm
from qiskit.quantum_info import SparsePauliOp
from sympy import simplify

class QuantumState:
    sparse_vector : Dict # map basis states to sympy-Symbolic
    substitutions: List = []

    def __init__(self, init_basis: Optional[int] = None, 
                 init_amplitude = complex(1.0, 0.0), qubits_used=[]):
        self.sparse_vector = dict()
        if init_basis is not None:
            self.insert_amplitude(init_basis, simplify(init_amplitude))
        if len(qubits_used) == 0:
            raise Exception("no indices of qubits specified!")
        self.qubits_used = sorted(qubits_used, reverse=False) # addresses of qubits that are used, we need this to build efficiently density matrices that contain only these qubits
        
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
        return isclose(self.get_amplitude(0), 1)
    
    def get_qubit_amplitudes(self) -> Tuple:
        return self.get_amplitude(0), self.get_amplitude(1)
    
    def insert_amplitude(self, basis: int, amplitude: Any) -> bool:
        amplitude = simplify(get_complex(amplitude))
        if isclose(amplitude.real, 0.0, abs_tol=Precision.isclose_abstol) and isclose(amplitude.imag, 0.0, abs_tol=Precision.isclose_abstol):
                return False

        self.sparse_vector[basis] = amplitude
        return True

    def add_amplitude(self, basis: int, amplitude: Any) ->  bool:
        amplitude = get_complex(amplitude)

        
        if isclose(amplitude.real, 0.0, abs_tol=Precision.isclose_abstol) and isclose(amplitude.imag, 0.0, abs_tol=Precision.isclose_abstol):
            # if both the real and the imaginary part of the amplitude are 0 then we return False because we do not add any amplitude
            return False
        
        prev_amplitude = self.get_amplitude(basis)
        current_amplitude = simplify(get_complex(prev_amplitude + amplitude))
        if isclose(current_amplitude.real, 0.0, abs_tol=Precision.isclose_abstol) and isclose(current_amplitude.imag, 0.0, abs_tol=Precision.isclose_abstol):
            # if the new amplitude is 0 in both the real and imaginary part we delete this key
            del self.sparse_vector[basis]
            return False
        self.sparse_vector[basis] = current_amplitude
        return True

    def normalize(self):
        sum_ = 0
        for val in self.sparse_vector.values():
            sum_ += val*conjugate(val)

        norm = simplify(sqrt(sum_))

        for key in self.sparse_vector.keys():
            val = self.sparse_vector[key] / norm
            self.sparse_vector[key] = simplify(val)

    def __eq__(self, other):
        assert isinstance(other, QuantumState)
        if len(self.sparse_vector.keys()) != len(other.sparse_vector.keys()):
            return False
        
        # here we check for global phases: two states are equal if they only differ by a global factor
        inner_product = get_fidelity(self, other)
        return isclose(abs(inner_product), 1) 
    
    def _get_physical_basis(self, virtual_basis: int):
        assert len(self.qubits_used) > 0
        vb_index = 0
        answer = ""
        for index in range(max(self.qubits_used)+1):
            if index == self.qubits_used[vb_index]:
                # if the current id(index) is being used then we set it to the value of virtual basis in that position (virtual_basis[index])
                if virtual_basis % 2 == 1:
                    answer += "1"
                else:
                    answer += "0"
                vb_index += 1    
                virtual_basis //= 2
            else:
                # otherwise we pad with zero
                answer += "0"
        return bin_to_int(answer)

    def get_density_matrix(self) -> List[List[int]]:
        result = []
        
        # initialize with zeros
        for _ in range(2**len(self.qubits_used)):
            temp = []
            for _ in range(2**len(self.qubits_used)):
                temp.append(0.0)
            result.append(temp)

        for virtual_row in range(2**len(self.qubits_used)): # the virtual row is only proportial to the size of the qubits that are actually used
            physical_row = self._get_physical_basis(virtual_row) # this is the real basis
            for virtual_col in range(2**len(self.qubits_used)):
                physical_col = self._get_physical_basis(virtual_col)
                assert result[virtual_row][virtual_col] == 0.0
                result[virtual_row][virtual_col] = self.get_amplitude(physical_row) * conjugate(self.get_amplitude(physical_col))

        assert len(result) == 2**len(self.qubits_used)
        return result
    
    @staticmethod
    def  _get_real_index(qubits_used, index):
        answer = None
        for (index_, q) in enumerate(qubits_used):
            if q == index:
                assert answer is None
                answer = index_
        assert answer is not None
        return answer
    
    def single_partial_trace(self, rho=None, index=0, qubits_used=None):
        """removes the qubits at `index` from the system

        Args:
            rho (_type_, optional): _description_. Defaults to None.
            index (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        if qubits_used is None:
            qubits_used = deepcopy(self.qubits_used)
        if index not in qubits_used:
            raise Exception(f"Cannot remove index {index} from quantum state with qubits{qubits_used}")

        
        if rho is None:
            rho = self.get_density_matrix()

        initial_dim = len(rho)
        assert (initial_dim % 2) == 0
        assert len(rho) == (2**(len(qubits_used)))

        # change index cause density matrix is only over qubits used
        index = QuantumState._get_real_index(qubits_used, index)

        # initialize the result as a matrix full of zeros. Since we are tracing out a qubit the new dimension is initial dimension/2
        result = []
        for _ in range(initial_dim//2):
            temp = []
            for _ in range(initial_dim//2):
                temp.append(0)                
            result.append(temp)
        for ket in range(initial_dim):
            bin_ket = int_to_bin(ket, zero_padding=initial_dim) # this is the original ket, we get the binary representation
            assert len(bin_ket) == initial_dim
            assert isinstance(bin_ket, str)
            bin_new_ket = bin_ket[:index] + bin_ket[index+1:] # now we create a binary string without the bit that we want to remove (located at index)
            assert len(bin_ket) == len(bin_new_ket) + 1
            index_new_ket = bin_to_int(bin_new_ket) # index of the row in the result(-ing density matrix)
            for bra in range(initial_dim):
                bin_bra = int_to_bin(bra, zero_padding=initial_dim) # original bra
                bin_new_bra = bin_bra[:index] + bin_bra[index+1:] # remove the bit in the index we dont want
                index_new_bra = bin_to_int(bin_new_bra) # index of the row in the result(-ing density matrix)   

                if bin_ket[index] == bin_bra[index]:
                    result[index_new_ket][index_new_bra] += rho[ket][bra]

        assert len(result) == initial_dim/2
        return result

    def multi_partial_trace(self, rho=None, remove_indices: List[int]=[0], qubits_used=None) -> List[List[float]]:
        if rho is None:
            rho = self.get_density_matrix()

        if qubits_used is None:
            qubits_used = deepcopy(self.qubits_used)

        assert len(rho) == 2**(len(qubits_used)) 

        remove_indices = sorted(remove_indices, reverse=True) # we remove higher indices first to avoid conflict
        for index in remove_indices:
            result = self.single_partial_trace(rho, index, qubits_used=qubits_used)
            qubits_used.remove(index)
        return result
    
    def get_trace(self, rho: List[List[float]]):
        result = 0
        for i in range(len(rho)):
            result += rho[i][i]
        return simplify(result)
    
    def to_np_array(self) -> np.array:
        result = []
        
        # initialize with zeros
        for _ in range(2**len(self.qubits_used)):
            result.append(0)
            
        for virtual_row in range(2**len(self.qubits_used)):
            physical_row = self._get_physical_basis(virtual_row)
            result[virtual_row] = self.get_amplitude(physical_row)
            
        return np.array(result)


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

def get_inner_product(qstate1, qstate2) -> float:
    inner_product = 0
    for (key, val1) in qstate1.sparse_vector.items():
        val2 = qstate2.get_amplitude(key)
        inner_product += val1*conjugate(val2)
    return simplify(inner_product)

def get_fidelity(qstate1: QuantumState, qstate2: QuantumState) -> float:
    inner_product = get_inner_product(qstate1, qstate2)
    return simplify(inner_product*conjugate(inner_product))


    
# np.array utils

def np_get_energy(H: SparsePauliOp, quantum_state: np.array) -> float:
    quantum_state = normalize_np_array(quantum_state)
    # Compute the expectation value ⟨q|H|q⟩
    q_dagger = np.conjugate(quantum_state).T  # Conjugate transpose of |q>
    H_q = np.dot(H, quantum_state)            # Matrix multiplication H|q>
    expectation_value = np.dot(q_dagger, H_q)  # Inner product ⟨q|H|q⟩
    assert isclose(expectation_value.imag, 0.0, abs_tol=Precision.isclose_abstol)
    return simplify(expectation_value.real)

def np_get_energy_from_rho(H: SparsePauliOp, density_matrix: np.array) -> float:
    trace_rho = np.trace(density_matrix)
    assert np.isclose(trace_rho, 1.0, atol=1e-9), "Density matrix trace is not 1"
    energy_matrix = np.matmul(density_matrix, H)
    
    energy = np.trace(energy_matrix)
    
    assert isclose(energy.imag, 0.0, abs_tol=Precision.isclose_abstol)
    return simplify(energy.real)



def np_schroedinger_equation(H: SparsePauliOp, t: complex, initial_state: np.array) -> np.array:
    I = complex(0, 1)

    # Compute the time evolution operator U = e^(iHt)
    U = expm(-I * H * t) # planck constant is assumed to be 1

    # Apply U to the initial state |q>
    final_state = np.dot(U, initial_state)
    
    # since U might not be a non-unitary matrix (for imaginary time evolution)
    final_state = normalize_np_array(final_state)
    return final_state

def np_get_fidelity(state1: np.array, state2: np.array) -> float:
    result = np.dot(np.conjugate(state1).T, state2 )
    return simplify(result.real*np.conjugate(result.real))

def normalize_np_array(quantum_state):
    # normalize quantum state
    norm = np.dot(np.conjugate(quantum_state).T, quantum_state)
    assert isclose(norm.imag, 0.0, abs_tol=Precision.isclose_abstol)
    norm = norm.real
    if not (norm > 0):
        raise Exception(f"Norm is 0: {norm}, for quantum state {quantum_state}")
    
    sq_norm = np.sqrt(norm)
    return simplify(quantum_state/sq_norm)

def np_array_to_qs(state_vector, qubits_used):
    qs = None
    for (index, amp) in enumerate(state_vector):
        if not isclose(amp, 0.0, abs_tol=Precision.isclose_abstol):
            if qs is None:
                qs = QuantumState(index, amp, qubits_used=qubits_used)
            else:
                assert qs.get_amplitude(index) == 0.0
                qs.insert_amplitude(index, amp) 

    return qs
