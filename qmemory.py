from cmath import cos, isclose, sin
from math import sqrt, pi
import math
import random
from typing import List, Optional

from numpy import conjugate

from ibm_noise_models import Instruction
from qstates import QuantumState, get_inner_product
from qpu_utils import *

from utils import get_kraus_matrix_probability, myceil, myfloor


def evaluate_op(op: Op, qubit: QuantumState, name: str, params=None, is_inverse: bool = False) -> Optional[QuantumState]:
    ''' Evaluates qubit->qubit op functions
    '''
    assert(qubit.is_qubit())
    result = QuantumState(qubits_used=qubit.qubits_used)
    a0, a1 = qubit.get_qubit_amplitudes()

    # some useful constants
    # half_prob = myfloor(complex(1/sqrt(2), 0.0), Precision.PRECISION)
    half_prob = complex(1/sqrt(2), 0.0)
    I = complex(0, 1)
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
        if isclose(prob, 0):
            return None
        result.insert_amplitude(0, new_a0)
        result.insert_amplitude(1, new_a1)
    elif op == Op.RX:
        assert params is not None
        assert not is_inverse
        assert len(params) == 1
        return evaluate_op(Op.U3, qubit, name, params=[params[0], 3*pi/2, pi/2], is_inverse=is_inverse)
    elif op == Op.RY:
        assert params is not None
        assert not is_inverse
        assert len(params) == 1
        return evaluate_op(Op.U3, qubit, name, params=[params[0], 0, 0], is_inverse=is_inverse)
    elif op == Op.RZ:
        assert params is not None
        assert not is_inverse
        assert len(params) == 1
        return evaluate_op(Op.U1, qubit, name, params=params)
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
    elif is_projector(op):
        assert not is_inverse
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
        return QuantumState(0, qubits_used=[address])
    else:
        return QuantumState(int(bin_number[address]), qubits_used=[address])
    
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
    result = QuantumState(qubits_used=quantum_state.qubits_used)
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
    controls = [gate_data.control]
    address = gate_data.address
    result = QuantumState(qubits_used=quantum_state.qubits_used)
    for (basis, value) in quantum_state.sparse_vector.items():
        if are_controls_true(basis, controls):
            basis_state = QuantumState(basis, value, qubits_used=quantum_state.qubits_used)
            if op == Op.CNOT:
                gate_data2 = GateData(Op.X, address, None)
                written_basis = write1(basis_state, gate_data2, is_inverse=is_inverse)
                assert written_basis is not None
            elif op == Op.CH:
                gate_data2 = GateData(Op.H, address, None)
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
            prob1 = myfloor(prob1, Precision.PRECISION)
        else:
            prob1 = myceil(prob1, Precision.PRECISION)
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


def get_seq_probability(quantum_state: QuantumState, seq: List[GateData], is_floor=True):
    count_meas = 0
    for s in seq:
        print(type(s))
        assert isinstance(s, GateData)
        assert s.label != Op.MEAS
        if s.label == Op.P0 or s.label == Op.P1:
            count_meas += 1
            if count_meas > 1:
                raise Exception("Invalid Measurement instruction")
        quantum_state = handle_write(quantum_state, s, is_inverse=False, normalize=False)
        if quantum_state is None:
            return None, 0.0

    prob = get_inner_product(quantum_state, quantum_state)
    
    if is_floor:
        if Precision.is_lowerbound:
            prob = myfloor(prob, Precision.PRECISION)
        else:
            prob = myceil(prob, Precision.PRECISION)
    else:
        prob = round(prob, Precision.PRECISION)
    quantum_state.normalize()
    return quantum_state, prob

def handle_write(quantum_state: QuantumState, gate_data: GateData, is_inverse=False, normalize=True):
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
            meas_instruction = Instruction(gate_data.address, Op.MEAS)
            q0, prob0 = get_seq_probability(quantum_state, [meas_instruction.get_gate_data(is_meas_0=True)])
            q1, prob1 = get_seq_probability(quantum_state, [meas_instruction.get_gate_data(is_meas_0=False)])
            result = random.choices([q0, q1], weights=[prob0, prob1], k=1)[0]
            if q1 is not None:
                if result == q1:
                    x_instruction = Instruction(gate_data.address, Op.X).get_gate_data()
                    result = write1(result, x_instruction)
        else:
            result = write1(quantum_state, gate_data, is_inverse=is_inverse)
    if normalize and (not (result is None)):
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