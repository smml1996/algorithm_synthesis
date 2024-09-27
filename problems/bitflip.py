from cmath import isclose
from copy import deepcopy
import os, sys
import time
from typing import Dict, List

from qiskit import QuantumCircuit
from cmemory import ClassicalState
from pomdp import build_pomdp
import qmemory
from qpu_utils import GateData, Op, Precision, BasisGates
from utils import are_matrices_equal
sys.path.append(os.getcwd()+"/..")

from qstates import QuantumState
from ibm_noise_models import Instruction, MeasChannel, NoiseModel, get_ibm_noise_model, HardwareSpec
import numpy as np
from math import pi   
from enum import Enum



POMDP_OUTPUT_DIR = "../synthesis/bitflip/"
WITH_TERMALIZATION = False
MAX_HORIZON = 7
MIN_HORIZON = 4
MAX_PRECISION = 30
TIME_OUT = 10800 # (in seconds) i.e 3 hours

class ExperimentID(Enum):
    IPMA = 0
    CXH = 1


class BitFlipInstance:
    def __init__(self,embedding: Dict[int, int]):
        """_summary_

        Args:
            num_qubits (int): _description_
            instruction_set (List[Instruction]): _description_
            embedding (Dict[int, int]): a mapping from logical qubits to physical qubits
        """
        self.embedding = embedding
        self.num_qubits = max(embedding.values())
        self.initial_state = None

        # check embedding
        assert 0 in self.embedding.keys()
        assert 1 in self.embedding.keys()
        assert 2 in self.embedding.keys()
        assert len(self.embedding.keys()) == 3

    def get_initial_states(self):
        """
        The initial state is specified as a uniform superpositions over all four bell states.
        """
        self.initial_state = []
        initial_cs = ClassicalState()

        H0 = Instruction(self.embedding[0], Op.H).get_gate_data()
        CX01 = Instruction(self.embedding[1], Op.CNOT, self.embedding[0]).get_gate_data()
        X0 = Instruction(self.embedding[0], Op.X).get_gate_data()
        Z0 = Instruction(self.embedding[0], Op.Z).get_gate_data()

        # prepare first bell state
        bell0 = QuantumState(0, dimension=self.num_qubits+1)
        bell0 = qmemory.handle_write(bell0, H0)
        bell0 = qmemory.handle_write(bell0, CX01)
        self.initial_state.append((bell0, initial_cs))

        # prepare second bell state
        bell1 = qmemory.handle_write(bell0, X0)
        self.initial_state.append((bell1, initial_cs))
    
        # prepare third bell state
        bell2 = qmemory.handle_write(bell0, Z0)
        self.initial_state.append((bell2, initial_cs))

        # preapre fourth bell state
        bell3 = qmemory.handle_write(bell2, X0)
        self.initial_state.append((bell3, initial_cs))


    def is_target_state(self, qs: QuantumState) -> bool:
        current_rho = qs.single_partial_trace(index=self.embedding[2])
        bell0_rho = self.initial_state[0].single_partial_trace(index=self.embedding[2])
        bell1_rho = self.initial_state[2].single_partial_trace(index=self.embedding[2])
        assert len(bell0_rho) == 4
        assert len(bell1_rho) == 4
        return are_matrices_equal(current_rho, bell0_rho) or are_matrices_equal(current_rho, bell1_rho)
    

# choosing embeddings
class ReadoutNoise:
    def __init__(self, target, success0, success1):
        self.target = target
        self.success0 = success0
        self.success1 = success1
        self.diff = success0 - success1
        self.acc_err = 1-success0 + 1-success1
        self.abs_diff = abs(success0-success1)

    def __str__(self):
        return f"{self.target}, {self.diff}, {self.acc_err}, {self.success0}, {self.success1}"
    def __repr__(self):
        return self.__str__()
    
def get_pivot_qubits(noise_model: NoiseModel):
    result = set()
    noises = []
    for qubit in range(noise_model.num_qubits):
        if noise_model.qubit_to_indegree[qubit] > 1:
            noise_data = noise_model.instructions_to_channel[Instruction(qubit, Op.MEAS)]
            assert isinstance(noise_data, MeasChannel)
            success0 = noise_data.get_ind_probability(0,0)
            success1 = noise_data.get_ind_probability(1,1)
            noises.append(ReadoutNoise(qubit, success0, success1))
    temp = sorted(noises, key=lambda x : x.success0)
    result.add(temp[0].target)
    result.add(temp[len(temp)-1].target)

    temp = sorted(noises, key=lambda x : x.success1)
    result.add(temp[0].target)
    result.add(temp[len(temp)-1].target)

    temp = sorted(noises, key=lambda x: x.acc_err) # accumulated error
    result.add(temp[0].target)
    result.add(temp[len(temp)-1].target)

    temp = sorted(noises, key=lambda x: x.diff)
    if temp[0].diff != temp[len(temp)-1].diff:
        result.add(temp[0].target)
        result.add(temp[len(temp)-1].target)

    temp = sorted(noises, key=lambda x: x.abs_diff)
    if temp[0].abs_diff != temp[len(temp)-1].abs_diff:
        result.add(temp[0].target)
        assert (temp[0].abs_diff < temp[len(temp)-1].abs_diff)
    return result

def get_selected_couplers(noise_model, target):
    couplers = noise_model.get_qubit_couplers(target)
    first_pair = (couplers[0], couplers[1]) # most noisy pair of couplers for this target
    second_pair = (couplers[len(couplers) -1], couplers[len(couplers) -2]) # least noisy pair of couplers for this target
    return [first_pair, second_pair]

def does_result_contains_d(result, d):
    for d_ in result:
        controls1 = set([d[0], d[1]])
        controls2 = set([d_[0], d_[1]])
        if d_[2] == d[2] and controls1 == controls2:
            return True
    return False

def get_backend_embeddings(backend: HardwareSpec):
    result = []
    noise_model = NoiseModel(backend, thermal_relaxation=WITH_TERMALIZATION)
    if noise_model.num_qubits < 14:
        pivot_qubits = set()
        for qubit in range(noise_model.num_qubits):
            if noise_model.get_qubit_indegree(qubit) > 1:
                pivot_qubits.add(qubit)
    else:
        pivot_qubits = get_pivot_qubits(noise_model)
    for target in pivot_qubits:
        assert(isinstance(target, int))
        for p in get_selected_couplers(noise_model, target):
            d_temp = dict()
            d_temp[0] = p[0][0]
            d_temp[1] = p[1][0]
            d_temp[2] = target
            if not does_result_contains_d(result, d_temp):
                result.append(deepcopy(d_temp))
    return result

class IBMBitFlipInstance:
    def __init__(self, embedding) -> None:
        self.embedding = embedding

    def prepare_bell_state(qc: QuantumCircuit, bell_index):
        qc.h(0)
        qc.cx(0, 1)
        if bell_index == 1:
            qc.x(0)
        elif bell_index == 2:
            qc.z(0)
        elif bell_index == 3:
            qc.x(0)
            qc.z(0)
        else:
            assert(bell_index == 0)

    def is_target_qs(self, state_vector):
        assert len(state_vector) == 8 # 3 qubits
        layout = [self.embedding[0], self.embedding[1], self.embedding[2]]

        qs = None

        for (index, amp) in state_vector:
            if not isclose(amp, 0.0, abs_tol=Precision.isclose_abstol):
                if qs is None:
                    qs = QuantumState(index, amp, dimension=8)
            pass

class Test:
    def __init__(self) -> None:
        pass
            
    @staticmethod
    def run_all():
        Test.check_bell_state_creation()

    @staticmethod
    def check_bell_state_creation():

        bf = BitFlipInstance({0:0, 1:1, 2:2})

        bf.get_initial_states()

        bell0_real_rho = [
                            [0.5, 0, 0, 0.5],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0.5, 0, 0, 0.5],
                          ]
        
        bell1_real_rho = [
                            [0, 0, 0, 0],
                            [0, 1, 1, 0],
                            [0, 1, 1, 0],
                            [0, 0, 0, 0],
                          ]
        
        bell2_real_rho = [
                            [0.5, 0, 0, -0.5],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [-0.5, 0, 0, 0.5],
                          ]
        
        bell3_real_rho = [
                            [0, 0, 0, 0],
                            [0, 1, -1, 0],
                            [0, -1, 1, 0],
                            [0, 0, 0, 0],
                          ]
        
        real_rhos = [bell0_real_rho, bell1_real_rho, bell2_real_rho, bell3_real_rho]
        for (index, initial_state) in bf.initial_state:
            assert isinstance(initial_state, QuantumState)
            current_rho = initial_state.single_partial_trace(index=2)
            assert are_matrices_equal(current_rho, real_rhos[index])

