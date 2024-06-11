from cmath import isclose
from math import sqrt, pi
import math
from typing import List, Optional
from sympy import *

from qstates import QuantumState
from utils import Gate, is_projection, GateData, Precision
from utils import myfloor, myceil


def is_multiqubit_gate(gate: Gate) -> bool:
    return gate in [Gate.CNOT, Gate.CU3, Gate.CZ]

def get_num_controls(gate: Gate) -> int:
    assert is_multiqubit_gate(gate)

    if gate == Gate.CNOT:
        return 1
    elif gate == Gate.CU3:
        return 1
    elif gate == Gate.CZ:
        return 1
    else:
        raise Exception(f"multiqubit gate not implemented ({gate})")


def evaluate_gate(gate: Gate, qubit: QuantumState, name: str, params=None, is_inverse: bool = True) -> Optional[QuantumState]:
    ''' Evaluates qubit->qubit gate functions
    '''
    assert(qubit.is_qubit())
    result = QuantumState()
    a0, a1 = qubit.get_qubit_amplitudes()

    # some useful constants
    # half_prob = myfloor(complex(1/sqrt(2), 0.0), Precision.PRECISION)
    half_prob = complex(1/sqrt(2), 0.0)
    if gate == Gate.I:
        return qubit
    elif gate == Gate.X:
        # X gate is its own inverse
        result.insert_amplitude(0, a1)
        result.insert_amplitude(1, a0)
    elif gate == Gate.Y:
        # Y gate is its own inverse
        result.insert_amplitude(0, I*a1)
        result.insert_amplitude(1, -I*a0)
    elif gate == Gate.Z:
        # Z gate is its own inverse
        result.insert_amplitude(0, a0)
        result.insert_amplitude(1, -a1)
    elif gate == Gate.H:
        # H gate is its own inverse
        result.insert_amplitude(0, half_prob * (a0 + a1))
        result.insert_amplitude(1, half_prob * (a0 - a1))
    elif gate == Gate.S:
        assert not is_inverse
        result.insert_amplitude(0, a0)
        result.insert_amplitude(1, I*a1)
    elif gate == Gate.SD:
        assert not is_inverse
        result.insert_amplitude(0, a0)
        result.insert_amplitude(1, -I*a1)
    elif gate == Gate.SX:
        assert not is_inverse
        # We implement RX gate which has only a global difference with SX
        result.insert_amplitude(0, half_prob* (a0 - I*a1))
        result.insert_amplitude(1, half_prob*(-I*a0  + a1))
    elif gate == Gate.SXD:
        assert not is_inverse
        result.insert_amplitude(0, half_prob * (a0 + a1*I))
        result.insert_amplitude(1, half_prob * (a0*I + a1))
    elif gate == Gate.TD:
        assert not is_inverse
        result.insert_amplitude(0, a0)
        # we have to calculate e^(-i*pi/4) = cos(-pi/4) + isin(-pi/4) = 1/sqrt{2} - i/sqrt{2}
        result.insert_amplitude(1, a1 * ( half_prob - (I * half_prob)))
    elif gate == Gate.T:
        assert not is_inverse
        result.insert_amplitude(0, a0)
        # we have to calculate e^(i*pi/4) = cos(pi/4) + isin(pi/4) = 1/sqrt{2} + i/sqrt{2}
        result.insert_amplitude(1, a1 * ( half_prob + (I * half_prob)))
    elif gate == Gate.U3:
        if is_inverse:
            raise Exception("Missing implementation of rever of gate U3")
        assert len(params) == 3
        cos_result = cos(params[0]/2)
        sin_result = sin(params[0]/2)
        result.insert_amplitude(0, (a0*cos_result)-
                                a1*(math.e**(I*params[2]))*sin_result)
        result.insert_amplitude(1, a0*(math.e**(I*params[1]))*sin_result+
                                a1*(math.e**(I*(params[1]+params[2])))*cos_result)
        
    elif gate == Gate.U2:
        if is_inverse:
            raise Exception("Missing implmentation of reverse of gate U2")
        
        assert len(params) == 2
        return evaluate_gate(Gate.U3, qubit, name, params=[pi/2.0, params[0], params[1]], is_inverse=is_inverse)
    elif gate == Gate.U1:
        if is_inverse:
            raise Exception("Missing implementation of reverse of gate U1")
        assert len(params) == 1
        return evaluate_gate(Gate.U3, qubit, name, params=[0,0,params[0]], is_inverse=is_inverse)
    elif gate == Gate.RESET:
        if is_inverse:
            raise Exception("Inverse of gate RESET not implemented.")
        else:
            assert a0 == 1.0 or a1 == 1.0
            result.insert_amplitude(0, 1.0)
    elif gate == Gate.XZ:
        if is_inverse:
            after_x = evaluate_gate(Gate.X, qubit, name)
            return evaluate_gate(Gate.Z, after_x, name)
        else:
            after_z = evaluate_gate(Gate.Z, qubit, name)
            return evaluate_gate(Gate.X, after_z, name)
    elif is_projection(gate):
        if is_inverse:
            s0 = Symbol(f"proj0_{name}")
            s1 = Symbol(f"proj1_{name}")
            QuantumState.symbolic_vars.append(s0)
            QuantumState.symbolic_vars.append(s1)
            result.insert_amplitude(0, s0)
            result.insert_amplitude(1, s1)

            if gate == Gate.P0:
                result.constraints.append(simplify(s0*conjugate(s0)) > 0)
            else:
                result.constraints.append(simplify(s1*conjugate(s1)) > 0)
        else:
            if gate == Gate.P0:
                if a0 == 0:
                    return None
                result.insert_amplitude(0, complex(1, 0.0))
            else:
                assert gate == Gate.P1
                if a1 == 0:
                    return None
                result.insert_amplitude(1, complex(1, 0.0))
    else:
        print(f"eval_gate: gate not implemented ({gate})")
    return result

def get_qubit_from_basis(basis: int, address: int) -> QuantumState:
    bin_number = "{0:b}".format(basis)[::-1]
    if address >= len(bin_number):
        return QuantumState(0)
    else:
        return QuantumState(int(bin_number[address]))
    
def glue_qubit_in_basis(basis: int, address: int, qubit_basis: int) -> int:
    bin_number: list(str) = list("{0:b}".format(basis)[::-1])
    # append zeros
    while len(bin_number) <= address:
        bin_number.append('0')
    # modify qubit in computational basis
    bin_number[address] = str(qubit_basis)
    # compute basis
    result: int = 0
    for b in bin_number[::-1]:
        result = (result << 1) + int(b)
    return result

def write1(quantum_state: QuantumState, gate_data: GateData, name="", is_inverse=False) -> Optional[QuantumState]:
    assert is_inverse == False # TODO: remove this
    result = QuantumState()
    at_least_one_perform_gate = False
    gate = gate_data.label
    address = gate_data.address
    for (basis, value) in quantum_state.sparse_vector.items():
        qubit = get_qubit_from_basis(basis, address)
        should_perform_gate = True
        if (not is_inverse) and is_projection(gate):
            if gate == Gate.P0 and (not qubit.is_qubit_0()):
                # we cannot apply a projectin to \0> if the qubit is 1 already
                should_perform_gate = False
            elif gate == Gate.P1 and qubit.is_qubit_0():
                should_perform_gate = False

        if should_perform_gate:
            qubit = evaluate_gate(gate, qubit, name, params=gate_data.params, is_inverse=is_inverse)
            a0, a1 = qubit.get_qubit_amplitudes()
            a0 *= value
            a1 *= value
            
            basis0 = glue_qubit_in_basis(basis, address, 0)
            basis1 = glue_qubit_in_basis(basis, address, 1)
            result.add_amplitude(basis0, a0)
            result.add_amplitude(basis1, a1)
            at_least_one_perform_gate = True
    if not at_least_one_perform_gate:
        return None 
    assert len(result.sparse_vector.keys()) > 0
    return result
    
def are_controls_true(basis, controls: List[int]) -> bool:
    bin_number: list(str) = list("{0:b}".format(basis)[::-1])
    for a in controls:
        if (a >= len(bin_number)) or (bin_number[a] == '0'):
            return False
    return True

def write2(quantum_state: QuantumState, gate_data: GateData, is_inverse: bool= False) -> Optional[QuantumState]:
    
    gate = gate_data.label
    controls = gate_data.controls
    address = gate_data.address
    result = QuantumState()
    # print(******)
    # print(quantum_state)
    for (basis, value) in quantum_state.sparse_vector.items():
        if are_controls_true(basis, controls):
            basis_state = QuantumState(basis, value)
            if gate == Gate.CNOT:
                gate_data2 = GateData(Gate.X, address, None)
                # print(basis_state, gate_data2)
                written_basis = write1(basis_state, gate_data2, is_inverse=is_inverse)
                assert written_basis is not None
            elif gate == Gate.CU3:
                gate_data2 = GateData(Gate.U3, address, None, gate_data.params)
                written_basis = write1(basis_state, gate_data2, is_inverse=is_inverse)
                # assert written_basis is not None
            elif gate == Gate.CZ:
                gate_data2 = GateData(Gate.Z, address, None)
                written_basis = write1(basis_state, gate_data2, is_inverse=is_inverse)
                assert written_basis is not None
            else:
                raise Exception(f"Multiqubit gate not defined ({gate})")
            if written_basis is not None:
                for (b, v) in written_basis.sparse_vector.items():
                    result.add_amplitude(b, v)
        else:
            result.add_amplitude(basis, value)
    assert len(result.sparse_vector.keys()) > 0
    return result


def get_qs_probability(quantum_state, address, is_zero=False, is_floor=True):
    prob1 = 0.0
    for (basis, value) in quantum_state.sparse_vector.items():
        if are_controls_true(basis, [address]):
            prob1 += value*conjugate(value)
    if is_floor:
        if Precision.is_lowerbound:
            prob1 = myfloor(simplify(prob1), Precision.PRECISION)
        else:
            prob1 = myceil(simplify(prob1), Precision.PRECISION)
    else:
        prob1 = round(prob1, Precision.PRECISION)
    if prob1 > 1.0:
        # assert isclose(prob1, 1.0, rel_tol=Precision.rel_tol)
        prob1 = 1.0
    else:
        assert prob1 <= 1.0
    if is_zero:
        return 1.0 - prob1
    else:
        return prob1


def get_seq_probability(quantum_state: QuantumState, seq):
    count_meas = 0
    for s in seq:
        assert isinstance(s, GateData)
        if s.label == Gate.P0 or s.label == Gate.P1:
            count_meas += 1
            if count_meas > 1:
                raise Exception("Invalid Measurement instruction")
    if count_meas > 0:
        for s in seq:
            assert isinstance(s, GateData)
            if s.label == Gate.P0:
                return get_qs_probability(quantum_state, s.address, is_zero = True)
            elif s.label == Gate.P1:
                return get_qs_probability(quantum_state, s.address, is_zero=False)
            quantum_state = handle_write(quantum_state, s, is_inverse=False)

    return 1.0

def handle_write(quantum_state: QuantumState, gate_data: GateData, is_inverse=False):
    gate = gate_data.label
    assert len(quantum_state.sparse_vector.keys()) > 0
    if is_multiqubit_gate(gate):
        result = write2(quantum_state, gate_data, is_inverse)
    else:
        if gate_data.label == Gate.RESET:
            result = write1(quantum_state, GateData(Gate.P0, gate_data.address, None), is_inverse=is_inverse)
            if result is None:
                result = write1(quantum_state, GateData(Gate.P1, gate_data.address, None), is_inverse=is_inverse)
                result = write1(quantum_state, GateData(Gate.X, gate_data.address, None), is_inverse=is_inverse)
        else:
            result = write1(quantum_state, gate_data, is_inverse=is_inverse)
    if not (result is None):
        result.remove_global_phases()
        result.normalize()
        assert len(result.sparse_vector.keys()) > 0
    return result
    
def get_qubit_amplitudes(qs: QuantumState, address: int):
    amplitude0 = 0.0
    amplitude1 = 0.0
    for (basis, value) in qs.sparse_vector.items():
        qubit = get_qubit_from_basis(basis, address)
        if qubit.is_qubit_0():
            amplitude0 += value
        else:
            amplitude1 += value
    return amplitude0, amplitude1

def get_qs_probabiltiies(quantum_state, address):
    prob1 = 0.0
    for (basis, value) in quantum_state.sparse_vector.items():
        if are_controls_true(basis, [address]):
            prob1 += value*conjugate(value)
    prob1 = simplify(prob1)
    if prob1 > 1.0:
        # assert isclose(prob1, 1.0)
        prob1 = 1.0
    else:
        assert prob1 <= 1.0
    return 1.0-prob1, prob1

def get_probabilities(quantum_state: QuantumState, address: int, seq):
    count_meas = 0
    for s in seq:
        assert isinstance(s, GateData)
        if s.label == Gate.P0 or s.label == Gate.P1:
            count_meas += 1
            if count_meas > 1:
                raise Exception("Invalid Measurement instruction")
    for s in seq:
        if s.label == Gate.P0 or s.label == Gate.P1:
            break
        quantum_state = handle_write(quantum_state, s, is_inverse=False)

    return get_qs_probabiltiies(quantum_state, address)