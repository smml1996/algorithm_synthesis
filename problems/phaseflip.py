import os, sys
sys.path.append(os.getcwd()+"/..")

from cmath import isclose
from copy import deepcopy
import time
from typing import Any, Dict, List
import json

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from algorithm import AlgorithmNode, execute_algorithm
from cmemory import ClassicalState
from pomdp import POMDP, POMDPAction, POMDPVertex, build_pomdp
import qmemory
from qpu_utils import GateData, Op, BasisGates
from utils import are_matrices_equal, find_enum_object, get_index, is_matrix_in_list, Precision
sys.path.append(os.getcwd()+"/..")

from qstates import QuantumState
from ibm_noise_models import Instruction, MeasChannel, NoiseModel, get_ibm_noise_model, HardwareSpec, ibm_simulate_circuit, load_config_file
import numpy as np
from math import ceil, pi   
from enum import Enum
from experiments_utils import PhaseflipExperimentID, ReadoutNoise, bitflips_guard, check_files, directory_exists, generate_configs, generate_diff_algorithms_file, generate_embeddings, generate_mc_guarantees_file, generate_pomdps, get_config_path, get_configs_path, get_embeddings_path, get_num_qubits_to_hardware, get_project_path, get_project_settings, bell_state_pts
import cProfile
import pstats

from bitflip import does_result_contains_d


WITH_TERMALIZATION = False
MAX_PRECISION = 10
TIME_OUT = 10800 # (in seconds) i.e 3 hours


class PhaseFlipInstance:
    def __init__(self, embedding):
        """_summary_

        Args:
            num_qubits (int): _description_
            instruction_set (List[Instruction]): _description_
            embedding (Dict[int, int]): a mapping from logical qubits to physical qubits
        """
        self.embedding = embedding
        self.initial_state = None
        self.get_initial_states()
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
        bell0 = QuantumState(0, qubits_used=list(self.embedding.values()))
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


    def get_reward(self, hybrid_state) -> float:
        qs , _ = hybrid_state
        current_rho = qs.single_partial_trace(index=self.embedding[2])
        initial_qs, _ = self.initial_state[0] 
        bell0_rho = initial_qs.single_partial_trace(index=self.embedding[2])
        initial_qs,_ = self.initial_state[1]
        bell1_rho = initial_qs.single_partial_trace(index=self.embedding[2])
        assert len(bell0_rho) == 4
        assert len(bell1_rho) == 4
        if are_matrices_equal(current_rho, bell0_rho) or are_matrices_equal(current_rho, bell1_rho):
            return 1.00
        else:
            return 0.00
    

# choosing embeddings
def get_pivot_qubits(noise_model: NoiseModel, experiment_id: PhaseflipExperimentID):
    result = set()
    noises = []
    if noise_model.hardware_spec == HardwareSpec.MELBOURNE:
        noise_model.num_qubits = 14
    for qubit in range(noise_model.num_qubits):
        if experiment_id == PhaseflipExperimentID.CXH:
            if noise_model.get_qubit_indegree(qubit) > 1:
                noise_data = noise_model.instructions_to_channel[Instruction(qubit, Op.MEAS)]
                assert isinstance(noise_data, MeasChannel)
                success0 = noise_data.get_ind_probability(0,0)
                success1 = noise_data.get_ind_probability(1,1)
                noises.append(ReadoutNoise(qubit, success0, success1))
        else:
            assert experiment_id == PhaseflipExperimentID.IPMA
            if noise_model.get_qubit_outdegree(qubit) > 1:
                noise_data = noise_model.instructions_to_channel[Instruction(qubit, Op.MEAS)]
                assert isinstance(noise_data, MeasChannel)
                success0 = noise_data.get_ind_probability(0,0)
                success1 = noise_data.get_ind_probability(1,1)
                noises.append(ReadoutNoise(qubit, success0, success1))

    temp = sorted(noises, key=lambda x : x.success0)
    result.add(temp[0].target)

    temp = sorted(noises, key=lambda x : x.success1)
    result.add(temp[0].target)

    temp = sorted(noises, key=lambda x: x.acc_err) # accumulated error
    result.add(temp[0].target)

    temp = sorted(noises, key=lambda x: x.diff)
    if temp[0].diff != temp[len(temp)-1].diff:
        result.add(temp[0].target)

    temp = sorted(noises, key=lambda x: x.abs_diff)
    if temp[0].abs_diff != temp[len(temp)-1].abs_diff:
        result.add(temp[0].target)
        assert (temp[0].abs_diff < temp[len(temp)-1].abs_diff)
    return result

def get_selected_couplers(noise_model, target):
    couplers = noise_model.get_qubit_couplers(target)
    first_pair = (couplers[0], couplers[1]) # most noisy pair of couplers for this target
    return first_pair

def get_hardware_embeddings(backend: HardwareSpec, experiment_id) -> List[Dict[int, int]]:
    result = []
    noise_model = NoiseModel(backend, thermal_relaxation=WITH_TERMALIZATION)
    assert noise_model.num_qubits >= 14
    pivot_qubits = get_pivot_qubits(noise_model, experiment_id)
    for target in pivot_qubits:
        assert(isinstance(target, int))
        p = get_selected_couplers(noise_model, target)
        d_temp = dict()
        d_temp[0] = p[0][0]
        d_temp[1] = p[1][0]
        d_temp[2] = target
        if not does_result_contains_d(result, d_temp):
            result.append(deepcopy(d_temp))
    return result

class IBMPhaseFlipInstance:
    SHOTS = 1024 * 4
    def __init__(self, embedding) -> None:
        self.embedding = embedding
        new_embedding = dict()
        values = sorted(self.embedding.values())
        for (key, value) in self.embedding.items():
            new_embedding[key] = get_index(value, values)
        self.phaseflip_instance = PhaseFlipInstance(new_embedding)

    @staticmethod
    def prepare_initial_state(qc: QuantumCircuit, bell_index: int):
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
        qs = None
        for (index, amp) in enumerate(state_vector):
            if not isclose(amp, 0.0, abs_tol=Precision.isclose_abstol):
                if qs is None:
                    qs = QuantumState(index, amp, qubits_used=list(self.embedding.keys()))
                else:
                    assert qs.get_amplitude(index) == 0.0
                    qs.insert_amplitude(index, amp) 
        return self.phaseflip_instance.is_target_qs((qs, None))

def get_experiments_actions(noise_model: NoiseModel, embedding: Dict[int,int], experiment_id: PhaseflipExperimentID):
    if experiment_id == PhaseflipExperimentID.IPMA:
        if noise_model.basis_gates in [BasisGates.TYPE1, BasisGates.TYPE6]:
            Z0 = POMDPAction("Z0", [Instruction(embedding[0], Op.U3, params=[0.0, 0.0, pi])])
        else:
            Z0 = POMDPAction("Z0", [Instruction(embedding[0], Op.SX), Instruction(embedding[0], Op.RZ, params=[pi]), Instruction(embedding[0], Op.SX)])
            
        if noise_model.basis_gates in [BasisGates.TYPE1, BasisGates.TYPE6]:
            H = POMDPAction("H", [
                Instruction(embedding[2], Op.U2, params=[0.0, pi]),
                Instruction(embedding[1], Op.CNOT, control=embedding[2]),
                Instruction(embedding[0], Op.CNOT, control=embedding[2]),
                Instruction(embedding[2], Op.U2, params=[0.0, pi])
                ])
        else:
            assert noise_model.basis_gates in [BasisGates.TYPE2, BasisGates.TYPE3, BasisGates.TYPE4, BasisGates.TYPE7]
            H = POMDPAction("H", [
                Instruction(embedding[2], Op.RZ, params=[pi/2]),
                Instruction(embedding[2], Op.SX),
                Instruction(embedding[2], Op.RZ, params=[pi/2]),
                Instruction(embedding[1], Op.CNOT, control=embedding[2]),
                Instruction(embedding[0], Op.CNOT, control=embedding[2]),
                Instruction(embedding[2], Op.RZ, params=[pi/2]),
                Instruction(embedding[2], Op.SX),
                Instruction(embedding[2], Op.RZ, params=[pi/2])
            ])
            
        P2 = POMDPAction("P2", [Instruction(embedding[2], Op.MEAS)])
        
        return [P2, Z0, H]
    else:
        assert experiment_id == PhaseflipExperimentID.CXH
        if noise_model.basis_gates in [BasisGates.TYPE1, BasisGates.TYPE6]:
            H0 = POMDPAction("H0", [Instruction(embedding[0], Op.U2, params=[0.0, pi])])
            H1 = POMDPAction("H1", [Instruction(embedding[1], Op.U2, params=[0.0, pi])])
        else:
            assert noise_model.basis_gates in [BasisGates.TYPE2, BasisGates.TYPE3, BasisGates.TYPE4, BasisGates.TYPE7]
            H0 = POMDPAction("H0", [
                Instruction(embedding[0], Op.RZ, params=[pi/2]),
                Instruction(embedding[0], Op.SX),
                Instruction(embedding[0], Op.RZ, params=[pi/2])
            ])
            H1 = POMDPAction("H1", [
                Instruction(embedding[1], Op.RZ, params=[pi/2]),
                Instruction(embedding[1], Op.SX),
                Instruction(embedding[1], Op.RZ, params=[pi/2])
            ])
        
        P2 = POMDPAction("P2", [Instruction(embedding[2], Op.MEAS)])
        CX21 = POMDPAction("CX21", [Instruction(embedding[1], Op.CNOT, control=embedding[2])])
        CX01 = POMDPAction("CX01", [Instruction(embedding[1], Op.CNOT, control=embedding[0])])
        return [H0, H1, CX21, CX01, P2]


if __name__ == "__main__":
    arg_backend = sys.argv[1]
    Precision.PRECISION = MAX_PRECISION
    Precision.update_threshold()
    settings = get_project_settings()
    project_path = settings["PROJECT_PATH"]
    
    allowed_hardware = []
    for hardware in HardwareSpec:
        noise_model = NoiseModel(hardware, thermal_relaxation=WITH_TERMALIZATION)
        if noise_model.num_qubits >= 14:
            allowed_hardware.append(hardware)
    if arg_backend == "gen_configs":
        # step 0
        generate_configs(experiment_id=PhaseflipExperimentID.IPMA, min_horizon=1, max_horizon=7, allowed_hardware=allowed_hardware)
        # generate_configs(experiment_id=PhaseflipExperimentID.CXH, min_horizon=4, max_horizon=7, allowed_hardware=allowed_hardware)
    elif arg_backend == "embeddings":
        # generate paper embeddings
        batches = get_num_qubits_to_hardware(WITH_TERMALIZATION, allowed_hardware=allowed_hardware)
        for num_qubits in batches.keys():
            ipma_config_path = get_config_path(PhaseflipExperimentID.IPMA, num_qubits)
            generate_embeddings(PhaseflipExperimentID.IPMA, num_qubits, get_hardware_embeddings)
            
            cxh_config_path = get_config_path( PhaseflipExperimentID.CXH, num_qubits)
            generate_embeddings(PhaseflipExperimentID.CXH, num_qubits, get_hardware_embeddings)
    elif arg_backend == "all_pomdps":
        # TODO: clean me up
        # step 2: generate all pomdps
        # config_path = sys.argv[2]
        # generate_pomdps(config_path)
        batches = get_num_qubits_to_hardware(WITH_TERMALIZATION, allowed_hardware=allowed_hardware)
        for num_qubits in batches.keys():
            generate_pomdps(PhaseflipExperimentID.IPMA, num_qubits, get_experiments_actions, PhaseFlipInstance, bitflips_guard)
    elif arg_backend == "mc_ipma":
        generate_mc_guarantees_file(PhaseflipExperimentID.IPMA, allowed_hardware, get_hardware_embeddings, get_experiments_actions, WITH_THERMALIZATION=WITH_TERMALIZATION) 
    elif arg_backend == "alg_ipma":
        generate_diff_algorithms_file(PhaseflipExperimentID.IPMA, allowed_hardware, get_hardware_embeddings, get_experiments_actions, with_thermalization=False)
    elif arg_backend == "test" :
        check_files(PhaseflipExperimentID.IPMA, allowed_hardware, with_thermalization=False)
        
    # step 3 synthesis of algorithms with C++ code and generate lambdas (guarantees)
    
        