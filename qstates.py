from typing import Any, List, Optional
from sympy import *
from qpu_utils import Precision, get_complex, int_to_bin
from math import isclose

class QuantumState:
    sparse_vector : Dict # map basis states to sympy-Symbolic
    substitutions: List = []

    def __init__(self, init_basis: Optional[int] = None, 
                 init_amplitude = complex(1.0, 0.0)):
        self.sparse_vector = dict()
        if not (init_basis is None):
            self.insert_amplitude(init_basis, init_amplitude)

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
    
    

def are_states_lists_equal(qsl1: List, qsl2: List) -> bool:    
    if len(qsl1) != len(qsl2):
        return False
    
    for qs1 in qsl1:
        if not (qs1 in qsl2):
            return False
    return True

def is_list_subset(subset: List, all_set: List) -> bool:
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

def is_bell_state(qs: QuantumState, address_space) -> bool:
    Precision.update_threshold()
    basis00 = 0.0
    basis11 = 0.0
    for (basis, value) in qs.sparse_vector.items():
        qubit0 = (basis >> address_space(0)) & 1
        qubit1 = (basis >> address_space(1)) & 1
        if qubit0 != qubit1:
            probv = value * conjugate(value)
            if isinstance(probv, complex):
                probv = probv.real
            if not isclose(probv, 0.0, abs_tol=Precision.isclose_abstol): 
                return False
        else:
            if qubit0 == 0:
                basis00 += value
            else:
                basis11 += value

    prob00 = basis00 * conjugate(basis00)

    if isinstance(prob00, complex):
        prob00 = prob00.real

    if isclose(prob00, 0.0, abs_tol=Precision.isclose_abstol):
        return False
    
    basis00 = complex(basis00)
    basis11 = complex(basis11)
    return isclose(basis00.real, basis11.real, rel_tol=Precision.rel_tol) and isclose(basis00.imag, basis11.imag, rel_tol=Precision.rel_tol)

def is_bell_state_minus(qs: QuantumState, address_space) -> bool:
    Precision.update_threshold()
    basis00 = 0.0
    basis11 = 0.0
    for (basis, value) in qs.sparse_vector.items():
        qubit0 = (basis >> address_space(0)) & 1
        qubit1 = (basis >> address_space(1)) & 1
        if qubit0 != qubit1:
            probv = value * conjugate(value)
            if isinstance(probv, complex):
                probv = probv.real
            if not isclose(probv, 0.0, abs_tol=Precision.isclose_abstol): 
                return False
        else:
            if qubit0 == 0:
                basis00 += value
            else:
                basis11 += value
    
    prob00 = basis00 * conjugate(basis00)

    if isinstance(prob00, complex):
        prob00 = prob00.real

    if isclose(prob00, 0.0, abs_tol=Precision.isclose_abstol):
        return False
    
    basis00 = complex(basis00)
    basis11 = complex(basis11)
    return isclose(basis00.real, -basis11.real, rel_tol=Precision.rel_tol) and isclose(basis00.imag, -basis11.imag, rel_tol=Precision.rel_tol)

def is_flip_bell_state(qs: QuantumState, address_space) -> bool:
    Precision.update_threshold()
    basis01 = 0.0
    basis10 = 0.0
    for (basis, value) in qs.sparse_vector.items():
        qubit0 = (basis >> address_space(0)) & 1
        qubit1 = (basis >> address_space(1)) & 1
        if qubit0 == qubit1:
            probv = value * conjugate(value)
            if isinstance(probv, complex):
                probv = probv.real
            if not isclose(probv, 0.0, abs_tol=Precision.isclose_abstol): 
                return False
        else:
            if qubit0 == 0:
                basis01 += value
            else:
                basis10 += value
    
    prob01 = basis01 * conjugate(basis01)

    if isinstance(prob01, complex):
        prob01 = prob01.real

    if isclose(prob01, 0.0, abs_tol=Precision.isclose_abstol):
        return False
    
    basis01 = complex(basis01)
    basis10 = complex(basis10)
    return isclose(basis01.real, basis10.real, rel_tol=Precision.rel_tol) and isclose(basis01.imag, basis10.imag, rel_tol=Precision.rel_tol)


def is_flip_bell_state_minus(qs: QuantumState, address_space) -> bool:
    Precision.update_threshold()
    basis01 = 0.0
    basis10 = 0.0
    for (basis, value) in qs.sparse_vector.items():
        qubit0 = (basis >> address_space(0)) & 1
        qubit1 = (basis >> address_space(1)) & 1
        if qubit0 == qubit1:
            probv = value * conjugate(value)
            if isinstance(probv, complex):
                probv = probv.real
            if not isclose(probv, 0.0, abs_tol=Precision.isclose_abstol): 
                return False
        else:
            if qubit0 == 0:
                basis01 += value
            else:
                basis10 += value
    
    prob01 = basis01 * conjugate(basis01)

    if isinstance(prob01, complex):
        prob01 = prob01.real

    if isclose(prob01, 0.0, abs_tol=Precision.isclose_abstol):
        return False
    
    basis01 = complex(basis01)
    basis10 = complex(basis10)
    return isclose(basis01.real, -basis10.real, rel_tol=Precision.rel_tol) and isclose(basis01.imag, -basis10.imag, rel_tol=Precision.rel_tol)

def is_ghz_state(qs: QuantumState, address_space):
    # we are looking for a superposition of |000> and |111>
    Precision.update_threshold()
    basis000 = 0.0
    basis111 = 0.0
    for (basis, value) in qs.sparse_vector.items():
        qubit0 = (basis >> address_space(0)) & 1
        qubit1 = (basis >> address_space(1)) & 1
        qubit2 = (basis >> address_space(2)) & 1
        if qubit0 == qubit1 and qubit1==qubit2:
            if qubit0 == 0:
                basis000 += value
            else:
                basis111 += value
        else:
            probv = value * conjugate(value)
            if isinstance(probv, complex):
                probv = probv.real
            if not isclose(probv, 0.0, abs_tol=Precision.isclose_abstol): 
                return False
            
    

    prob000 = basis000 * conjugate(basis000)

    if isinstance(prob000, complex):
        prob000 = prob000.real

    if isclose(prob000, 0.0, abs_tol=Precision.isclose_abstol):
        return False
    
    basis000 = complex(basis000)
    basis111 = complex(basis111)
    return isclose(basis000.real, basis111.real, rel_tol=Precision.rel_tol) and isclose(basis000.imag, basis111.imag, rel_tol=Precision.rel_tol)

def is_quantum_state_entangled(qs, address_space):
    Precision.update_threshold()
    prob00 = 0.0
    prob11 = 0.0
    prob01 = 0.0
    prob10 = 0.0
    for (basis, value) in qs.sparse_vector.items():
        qubit0 = (basis >> address_space[0]) & 1
        qubit1 = (basis >> address_space[1]) & 1
        if qubit0 != qubit1:
            probv = value * conjugate(value)
            if isinstance(probv, complex):
                probv = probv.real
            if qubit0 == 0:
                prob10 += probv
            else:
                prob01 += probv
        else:
            if qubit0 == 0:
                prob00 += value * conjugate(value)
            else:
                prob11 += value * conjugate(value)

    if isinstance(prob00, complex):
        prob00 = prob00.real

    if isinstance(prob11, complex):
        prob11 = prob11.real
    if isclose(prob00, 0.0, abs_tol=Precision.isclose_abstol):
        return isclose(prob01, prob10)

    return isclose(prob00, prob11)

def get_events(all_events: List, qs: QuantumState):
    assert isinstance(qs, QuantumState)
    result = []
    for (qs_, event_name) in all_events:
        if qs == qs_:
            result.append(event_name)
    return result