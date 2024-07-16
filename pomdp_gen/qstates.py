from typing import Any, List, Optional
from sympy import *
from utils import Precision, int_to_bin, myfloor, myceil, get_mapped_basis
from math import isclose
import numpy as np

# TODO: add constraints
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

    def make_substitutions(self, expr):
        new_expr = expr
        for (e, val) in QuantumState.substitutions:
            try:
                new_expr = new_expr.subs(e, val)
            except:
                continue
        return new_expr
    
    def truncate(self):
        keys_to_remove = []
        for (key, val_) in self.sparse_vector.items():
            if Precision.is_lowerbound:
                val = myfloor(val_, Precision.PRECISION)
            else:
                val = myceil(val_, Precision.PRECISION)
            if not isclose(val, 0.0, abs_tol=Precision.isclose_abstol):
                self.sparse_vector[key] = val
            else:
                keys_to_remove.append(key)
        for k in keys_to_remove:
            del self.sparse_vector[k]

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
    
    def get_density_matrix(self, address_space, num_qubits=3) -> List[List[int]]:
        # TODO: generalize this
        result = []
        for i in range(2**num_qubits):
            result.append([0 for _ in range(2**num_qubits)])

        for i in range(2**num_qubits):
            for j in range(2**num_qubits):
                result[i][j] = self.get_amplitude(get_mapped_basis(i, address_space)) * conjugate(self.get_amplitude(get_mapped_basis(j, address_space)))
        return np.array(result)
        
    def is_qubit_0(self) -> bool:
        assert self.is_qubit()
        return isclose(simplify(self.get_amplitude(0)), 1)
    
    def get_qubit_amplitudes(self) -> Tuple:
        return self.get_amplitude(0), self.get_amplitude(1)
    
    def insert_amplitude(self, basis: int, amplitude: Any) -> bool:
        Precision.update_threshold()
        try:
            if not isinstance(amplitude, complex):
                amplitude = complex(amplitude, 0.0)
            if isinstance(amplitude, complex):
                if isclose(amplitude.real, 0.0, abs_tol=Precision.isclose_abstol):
                    amplitude = complex(0.0, amplitude.imag)
                if isclose(amplitude.imag, 0.0, abs_tol=Precision.isclose_abstol):
                    amplitude = complex(amplitude.real, 0.0)
                if isclose(amplitude.real, 0.0, abs_tol=Precision.isclose_abstol) and isclose(amplitude.imag, 0.0, abs_tol=Precision.isclose_abstol):
                    return False
        except:
            pass
        
        val = simplify(self.make_substitutions(amplitude))
        self.sparse_vector[basis] = simplify(self.make_substitutions(amplitude))
        return True

    def add_amplitude(self, basis: int, amplitude: Any) ->  bool:
        Precision.update_threshold()
        amplitude = simplify(amplitude)
        if isinstance(amplitude, Number):
            if not isinstance(amplitude, complex):
                amplitude = complex(amplitude, 0.0)

            if isinstance(amplitude, complex):
                if isclose(amplitude.real, 0.0, abs_tol=Precision.isclose_abstol):
                    amplitude = complex(0, amplitude.imag)
                if isclose(amplitude.imag, 0.0, abs_tol=Precision.isclose_abstol):
                    amplitude = complex(amplitude.real, 0.0)
                if isclose(amplitude.real, 0.0, abs_tol=Precision.isclose_abstol) and isclose(amplitude.imag, 0.0, abs_tol=Precision.isclose_abstol):
                    return False
        prev_amplitude = self.get_amplitude(basis)
        current_amplitude = simplify(prev_amplitude + amplitude)

        if isinstance(current_amplitude, Number):
            if not isinstance(current_amplitude, complex):
                current_amplitude = complex(current_amplitude, 0.0)
            
            if isinstance(current_amplitude, complex):
                if isclose(current_amplitude.real, 0.0, abs_tol=Precision.isclose_abstol) and isclose(current_amplitude.imag, 0.0, abs_tol=Precision.isclose_abstol):
                    del self.sparse_vector[basis]
                    return False
        self.sparse_vector[basis] = self.make_substitutions(current_amplitude)
        # assert(self.get_amplitude(basis) == current_amplitude)
        return True

    def normalize(self):
        # self.truncate()
        sum_ = 0
        for val in self.sparse_vector.values():
            sum_ += simplify(val*conjugate(val))

        norm = simplify(sqrt(sum_))

        for key in self.sparse_vector.keys():
            val = simplify(self.make_substitutions(self.sparse_vector[key] / norm))
            self.sparse_vector[key] = val
        # self.truncate()

    

    def __eq__(self, other):
        if len(self.sparse_vector.keys()) != len(other.sparse_vector.keys()):
            return False
        def check_equality_up_to_global_factor(factor):
            for (key, val) in self.sparse_vector.items():
                other_val_ = other.get_amplitude(key)
                other_val = factor*other_val_

                try:
                    if not isinstance(val, complex):
                        val = complex(val, 0.0)
                    if not isinstance(other_val, complex):
                        other_val = complex(other_val, 0.0)
                    if (not isclose(val.real, other_val.real, rel_tol=Precision.rel_tol)) or (not isclose(val.imag, other_val.imag, rel_tol=Precision.rel_tol)):
                        return False
                except:
                    if val != other_val:
                        return False
            return True
        if check_equality_up_to_global_factor(1):
            return True
        if check_equality_up_to_global_factor(-1):
            return True
        if check_equality_up_to_global_factor(I):
            return True
        if check_equality_up_to_global_factor(-I):
            return True
        return False
    
    def remove_global_phases(self):
        real_part = None
        complex_part = None
        have_all_same_abs = True
        for val in self.sparse_vector.values():
            
            if not isinstance(val, complex):
                try:
                    val = complex(val, 0.0)
                except:
                    have_all_same_abs = False
                    break

            # elif not isinstance(val, complex):
            #     have_all_same_abs = False
            #     break

            if real_part is None:
                real_part = abs(val.real)
            elif real_part != abs(val.real):
                have_all_same_abs = False
                break

            
            if complex_part is None:
                complex_part = abs(val.imag)
            elif complex_part != abs(val.imag):
                have_all_same_abs = False
            
        if have_all_same_abs:
            if complex_part == 0.0:
                assert real_part is not None
                for key in self.sparse_vector.keys():
                    self.sparse_vector[key] =simplify(self.sparse_vector[key]/real_part)
            elif real_part == 0.0:
                for key in self.sparse_vector.keys():
                    self.sparse_vector[key]= simplify(self.sparse_vector[key]/complex_part)
            else:
                factor = complex(real_part, complex_part)
                for key in self.sparse_vector.keys():
                    if self.sparse_vector[key] == factor:
                        self.sparse_vector[key] = 1
                    else:
                        # print(self.sparse_vector[key], factor)
                        # assert self.sparse_vector[key] == factor.conjugate()
                        self.sparse_vector[key] = simplify(self.sparse_vector[key]*factor.conjugate()/ (factor*factor.conjugate()))

        are_all_negative = True

        for val in self.sparse_vector.values():
            if not isinstance(val, complex):
                try:
                    val = complex(val, 0.0)
                except:
                    are_all_negative = False
                    break
            # elif not isinstance(val, complex):
            #     are_all_negative = False
            #     break
            if val.real > 0 or val.imag > 0:
                are_all_negative = False
                break
        if are_all_negative:
            for key in self.sparse_vector.keys():
                self.sparse_vector[key] *=-1

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