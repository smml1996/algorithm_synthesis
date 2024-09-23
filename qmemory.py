from cmath import isclose
from math import sqrt, pi
import math
from typing import List, Optional
from sympy import *

from qstates import QuantumState
from qpu_utils import *
from copy import deepcopy

from utils import get_kraus_matrix_probability


def evaluate_op(op: Op, qubit: QuantumState, name: str, params=None, is_inverse: bool = True) -> Optional[QuantumState]:
    ''' Evaluates qubit->qubit op functions
    '''
    assert(qubit.is_qubit())
    result = QuantumState()
    a0, a1 = qubit.get_qubit_amplitudes()

    # some useful constants
    # half_prob = myfloor(complex(1/sqrt(2), 0.0), Precision.PRECISION)
    half_prob = complex(1/sqrt(2), 0.0)
    if op == Op.I:
        return qubit
    elif op == Op.X:
        # X op is its own inverse
        result.insert_amplitude(0, a1)
        result.insert_amplitude(1, a0)
    elif op == Op.Y:
        # Y op is its own inverse
        result.insert_amplitude(0, I*a1)
        result.insert_amplitude(1, -I*a0)
    elif op == Op.Z:
        # Z op is its own inverse
        result.insert_amplitude(0, a0)
        result.insert_amplitude(1, -a1)
    elif op == Op.H:
        # H op is its own inverse
        result.insert_amplitude(0, half_prob * (a0 + a1))
        result.insert_amplitude(1, half_prob * (a0 - a1))
    elif op == Op.S:
        assert not is_inverse
        result.insert_amplitude(0, a0)
        result.insert_amplitude(1, I*a1)
    elif op == Op.SD:
        assert not is_inverse
        result.insert_amplitude(0, a0)
        result.insert_amplitude(1, -I*a1)
    elif op == Op.SX:
        assert not is_inverse
        # We implement RX op which has only a global difference with SX
        result.insert_amplitude(0, half_prob* (a0 - I*a1))
        result.insert_amplitude(1, half_prob*(-I*a0  + a1))
    elif op == Op.SXD:
        assert not is_inverse
        result.insert_amplitude(0, half_prob * (a0 + a1*I))
        result.insert_amplitude(1, half_prob * (a0*I + a1))
    elif op == Op.TD:
        assert not is_inverse
        result.insert_amplitude(0, a0)
        # we have to calculate e^(-i*pi/4) = cos(-pi/4) + isin(-pi/4) = 1/sqrt{2} - i/sqrt{2}
        result.insert_amplitude(1, a1 * ( half_prob - (I * half_prob)))
    elif op == Op.T:
        assert not is_inverse
        result.insert_amplitude(0, a0)
        # we have to calculate e^(i*pi/4) = cos(pi/4) + isin(pi/4) = 1/sqrt{2} + i/sqrt{2}
        result.insert_amplitude(1, a1 * ( half_prob + (I * half_prob)))
    elif op == Op.U3:
        if is_inverse:
            raise Exception("Missing implementation of rever of op U3")
        assert len(params) == 3
        cos_result = cos(params[0]/2)
        sin_result = sin(params[0]/2)
        result.insert_amplitude(0, (a0*cos_result)-
                                a1*(math.e**(I*params[2]))*sin_result)
        result.insert_amplitude(1, a0*(math.e**(I*params[1]))*sin_result+
                                a1*(math.e**(I*(params[1]+params[2])))*cos_result)
    elif op == Op.CUSTOM:
        assert params is not None
        assert not is_inverse
        prob, new_a0, new_a1 = get_kraus_matrix_probability(params, a0, a1, return_new_ampl=True)
        if isclose(simplify(prob), 0):
            return None
        result.insert_amplitude(0, new_a0)
        result.insert_amplitude(1, new_a1)
    elif op == Op.U2:
        if is_inverse:
            raise Exception("Missing implmentation of reverse of op U2")
        
        assert len(params) == 2
        return evaluate_op(Op.U3, qubit, name, params=[pi/2.0, params[0], params[1]], is_inverse=is_inverse)
    elif op == Op.U1:
        if is_inverse:
            raise Exception("Missing implementation of reverse of op U1")
        assert len(params) == 1
        return evaluate_op(Op.U3, qubit, name, params=[0,0,params[0]], is_inverse=is_inverse)
    elif op == Op.RESET:
        if is_inverse:
            raise Exception("Inverse of op RESET not implemented.")
        else:
            assert a0 == 1.0 or a1 == 1.0
            result.insert_amplitude(0, 1.0)
    elif op == Op.XZ:
        if is_inverse:
            after_x = evaluate_op(Op.X, qubit, name)
            return evaluate_op(Op.Z, after_x, name)
        else:
            after_z = evaluate_op(Op.Z, qubit, name)
            return evaluate_op(Op.X, after_z, name)
    elif is_projector(op):
        if is_inverse:
            s0 = Symbol(f"proj0_{name}")
            s1 = Symbol(f"proj1_{name}")
            QuantumState.symbolic_vars.append(s0)
            QuantumState.symbolic_vars.append(s1)
            result.insert_amplitude(0, s0)
            result.insert_amplitude(1, s1)

            if op == Op.P0:
                result.constraints.append(simplify(s0*conjugate(s0)) > 0)
            else:
                result.constraints.append(simplify(s1*conjugate(s1)) > 0)
        else:
            if op == Op.P0:
                if a0 == 0:
                    return None
                result.insert_amplitude(0, complex(1, 0.0))
            else:
                assert op == Op.P1
                if a1 == 0:
                    return None
                result.insert_amplitude(1, complex(1, 0.0))
    else:
        print(f"eval_op: op not implemented ({op})")
    return result

def get_qubit_from_basis(basis: int, address: int) -> QuantumState:
    bin_number = "{0:b}".format(basis)[::-1]
    if address >= len(bin_number):
        return QuantumState(0)
    else:
        return QuantumState(int(bin_number[address]))
    
def glue_qubit_in_basis(basis: int, address: int, qubit_basis: int) -> int:
    bin_number = list("{0:b}".format(basis)[::-1])
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
    at_least_one_perform_op = False
    op = gate_data.label
    address = gate_data.address
    for (basis, value) in quantum_state.sparse_vector.items():
        qubit = get_qubit_from_basis(basis, address)
        should_perform_op = True
        if (not is_inverse) and is_projector(op):
            if op == Op.P0 and (not qubit.is_qubit_0()):
                # we cannot apply a projectin to \0> if the qubit is 1 already
                should_perform_op = False
            elif op == Op.P1 and qubit.is_qubit_0():
                should_perform_op = False

        if should_perform_op:
            qubit = evaluate_op(op, qubit, name, params=gate_data.params, is_inverse=is_inverse)
            a0, a1 = qubit.get_qubit_amplitudes()
            a0 *= value
            a1 *= value
            
            basis0 = glue_qubit_in_basis(basis, address, 0)
            basis1 = glue_qubit_in_basis(basis, address, 1)
            result.add_amplitude(basis0, a0)
            result.add_amplitude(basis1, a1)
            at_least_one_perform_op = True
    if not at_least_one_perform_op:
        return None 
    assert len(result.sparse_vector.keys()) > 0
    return result
    
def are_controls_true(basis, controls: List[int]) -> bool:
    bin_number = list("{0:b}".format(basis)[::-1])
    for a in controls:
        if (a >= len(bin_number)) or (bin_number[a] == '0'):
            return False
    return True

def write2(quantum_state: QuantumState, gate_data: GateData, is_inverse: bool= False) -> Optional[QuantumState]:
    
    op = gate_data.label
    controls = gate_data.controls
    address = gate_data.address
    result = QuantumState()
    # print(******)
    # print(quantum_state)
    for (basis, value) in quantum_state.sparse_vector.items():
        if are_controls_true(basis, controls):
            basis_state = QuantumState(basis, value)
            if op == Op.CNOT:
                gate_data2 = GateData(Op.X, address, None)
                # print(basis_state, gate_data2)
                written_basis = write1(basis_state, gate_data2, is_inverse=is_inverse)
                assert written_basis is not None
            elif op == Op.CU3:
                gate_data2 = GateData(Op.U3, address, None, gate_data.params)
                written_basis = write1(basis_state, gate_data2, is_inverse=is_inverse)
                # assert written_basis is not None
            elif op == Op.CZ:
                gate_data2 = GateData(Op.Z, address, None)
                written_basis = write1(basis_state, gate_data2, is_inverse=is_inverse)
                assert written_basis is not None
            else:
                raise Exception(f"Multiqubit op not defined ({op})")
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
        if s.label == Op.P0 or s.label == Op.P1:
            count_meas += 1
            if count_meas > 1:
                raise Exception("Invalid Measurement instruction")
    if count_meas > 0:
        for s in seq:
            assert isinstance(s, GateData)
            if s.label == Op.P0:
                return get_qs_probability(quantum_state, s.address, is_zero = True)
            elif s.label == Op.P1:
                return get_qs_probability(quantum_state, s.address, is_zero=False)
            quantum_state = handle_write(quantum_state, s, is_inverse=False)

    return 1.0

def handle_write(quantum_state: QuantumState, gate_data: GateData, is_inverse=False):
    op = gate_data.label
    assert len(quantum_state.sparse_vector.keys()) > 0
    if is_multiqubit_gate(op):
        if op == Op.SWAP:
            index1 = gate_data.target
            assert len(gate_data.controls) == 1
            index2 = gate_data.controls[0]
            
            cx_gate1 = GateData(Op.CNOT, index2, controls=[index1])
            cx_gate2 = GateData(Op.CNOT, index1, controls=[index2])
            result = handle_write(quantum_state, cx_gate1)
            result = handle_write(quantum_state, cx_gate2)
            result = handle_write(quantum_state, cx_gate1)
        else:
            result = write2(quantum_state, gate_data, is_inverse)
    else:
        if gate_data.label == Op.RESET:
            result = write1(quantum_state, GateData(Op.P0, gate_data.address, None), is_inverse=is_inverse)
            if result is None:
                result = write1(quantum_state, GateData(Op.P1, gate_data.address, None), is_inverse=is_inverse)
                result = write1(quantum_state, GateData(Op.X, gate_data.address, None), is_inverse=is_inverse)
        else:
            result = write1(quantum_state, gate_data, is_inverse=is_inverse)
    if not (result is None):
        result.remove_global_phases()
        result.normalize()
        assert len(result.sparse_vector.keys()) > 0
    return result

def permute(quantum_state: QuantumState, desired_position: List[int], current_position = List[int]):
    """_summary_

    Args:
        quantum_state (QuantumState): _description_
        desired_position (List[]): list of physical addresses, how we want qubits to be arranged
        current_mapping (Dict[int, int]): list of physical addresses, how are qubits currently arranged
    """

    def get_current_index(val):
        for (index, curr_v) in enumerate(current_position):
            if curr_v == val:
                return index
        raise Exception("value not found")

    for (index, desired_x) in enumerate(desired_position): 
        if current_position[index] != desired_x:
            curr_index = get_current_index(desired_x)
            gate_data = GateData(Op.SWAP, index, [curr_index])
            quantum_state = handle_write(quantum_state, gate_data)

            # update current position
            temp = current_position[curr_index]
            current_position[curr_index] = desired_position[index]
            current_position[index] = temp

    assert desired_position == current_position
    return quantum_state
    
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
        if s.label == Op.P0 or s.label == Op.P1:
            count_meas += 1
            if count_meas > 1:
                raise Exception("Invalid Measurement instruction")
    for s in seq:
        if s.label == Op.P0 or s.label == Op.P1:
            break
        quantum_state = handle_write(quantum_state, s, is_inverse=False)

    return get_qs_probabiltiies(quantum_state, address)