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
from pomdp import POMDP, POMDPAction, POMDPVertex, build_pomdp, default_guard
import qmemory
from qpu_utils import GateData, Op, BasisGates
from utils import are_matrices_equal, find_enum_object, get_index, is_matrix_in_list, Precision
sys.path.append(os.getcwd()+"/..")

from qstates import QuantumState
from ibm_noise_models import Instruction, MeasChannel, NoiseModel, get_ibm_noise_model, HardwareSpec, ibm_simulate_circuit, load_config_file
import numpy as np
from math import pi   
from enum import Enum
from experiments_utils import GHZExperimentID, ReadoutNoise, default_load_embeddings, directory_exists, generate_configs, generate_embeddings, generate_pomdps, get_config_path, get_embeddings_path, get_num_qubits_to_hardware, get_project_path, get_project_settings
from bitflip import does_result_contains_d

WITH_TERMALIZATION = False
MAX_PRECISION = 10    

class GHZInstance:
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
        self.target_state = None
        self.set_target_state()
        
    def get_initial_states(self):
        self.initial_state = [(QuantumState(0, qubits_used=list(self.embedding.values())),ClassicalState())]
    
    def set_target_state(self):
        H0 = Instruction(self.embedding[0], Op.H).get_gate_data()
        CX01 = Instruction(self.embedding[1], Op.CNOT, self.embedding[0]).get_gate_data()
        CX12 = Instruction(self.embedding[2], Op.CNOT, self.embedding[1]).get_gate_data()
        qs = QuantumState(0, qubits_used=list(self.embedding.values()))
        qs = qmemory.handle_write(qs, H0)
        qs = qmemory.handle_write(qs, CX01)
        qs = qmemory.handle_write(qs, CX12)
        self.target_state = qs
        
    
    def get_reward(self, hybrid_state) -> float:
        qs, _ = hybrid_state
        if qs == self.target_state:
            return 1.0
        else:
            return 0.0
        
# choosing embeddings

def get_selected_couplers(noise_model, target):
    couplers = noise_model.get_qubit_couplers(target)
    first_pair = (couplers[0], couplers[1]) # most noisy pair of couplers for this target
    return first_pair

def get_valid_third(noise_model, coupler) -> bool:
    control = coupler[0]
    target = coupler[1] 
    for q in [control, target]:
        for is_target in [True, False]:
            couplers = noise_model.get_qubit_couplers(q, is_target=is_target)
            for (third_q, _) in couplers:
                if third_q != control and third_q != target:
                    return third_q
    return None
    
def is_repeated_embedding(all_embeddings, current) -> bool:
    current_set = set()
    for value in current.values():
        current_set.add(value)
    for embedding in all_embeddings:
        temp_s = set()
        for v in embedding.values():
            temp_s.add(v)
        if temp_s == current_set:
            return True
    return False
     
def get_hardware_embeddings(backend: HardwareSpec, experiment_id) -> List[Dict[int, int]]:
    result = []
    noise_model = NoiseModel(backend, thermal_relaxation=WITH_TERMALIZATION)
    
    if experiment_id == GHZExperimentID.EXP1:
        assert noise_model.num_qubits >= 14
        couplers = noise_model.get_most_noisy_couplers()
        for (coupler, prob_) in couplers:
            if len(result) == 3:
                break
            third_qubit = get_valid_third(noise_model, coupler)
            if third_qubit is not None:
                d_temp = dict()
                d_temp[0] = coupler[0]
                d_temp[1] = coupler[1]
                d_temp[2] = third_qubit
                assert third_qubit != coupler[0]
                assert third_qubit != coupler[1]
                assert coupler[0] != coupler[1]
                if not is_repeated_embedding(result, d_temp):
                    result.append(deepcopy(d_temp))
        return result   
    else:
        raise Exception("Implement me")


def get_experiments_actions(noise_model: NoiseModel, embedding: Dict[int,int], experiment_id: GHZExperimentID):
    if experiment_id == GHZExperimentID.EXP1:
        if noise_model.basis_gates in [BasisGates.TYPE1, BasisGates.TYPE6]:
            H0 = POMDPAction("H0", [Instruction(embedding[0], Op.U2, params=[0.0, pi])])
            H1 = POMDPAction("H1", [Instruction(embedding[1], Op.U2, params=[0.0, pi])])
            H2 = POMDPAction("H2", [Instruction(embedding[2], Op.U2, params=[0.0, pi])])
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
            H2 = POMDPAction("H2", [
                Instruction(embedding[2], Op.RZ, params=[pi/2]),
                Instruction(embedding[2], Op.SX),
                Instruction(embedding[2], Op.RZ, params=[pi/2])
            ])
            
        answer = [H0, H1, H2]
        
        for (v_control, control) in embedding.items():
            for (v_target, target) in embedding.items():
                if control!= target:
                    instruction = Instruction(target, Op.CNOT, control=control)
                    if instruction in noise_model.instructions_to_channel.keys():
                        action = POMDPAction(f"CX{v_control}{v_target}", [instruction])
                        answer.append(action)
        return answer
    else:
        raise Exception(f"No channels specified for experiment {experiment_id}")

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
        generate_configs(experiment_id=GHZExperimentID.EXP1, min_horizon=3, max_horizon=4, allowed_hardware=allowed_hardware)
    elif arg_backend == "embeddings":
        # generate paper embeddings
        batches = get_num_qubits_to_hardware(WITH_TERMALIZATION, allowed_hardware=allowed_hardware)
        for num_qubits in batches.keys():
            generate_embeddings(GHZExperimentID.EXP1, num_qubits, get_hardware_embeddings=get_hardware_embeddings)
    elif arg_backend == "all_pomdps":
        batches = get_num_qubits_to_hardware(WITH_TERMALIZATION, allowed_hardware=allowed_hardware)
        for num_qubits in batches.keys():
            generate_pomdps(GHZExperimentID.EXP1, num_qubits, get_experiments_actions, GHZInstance)

        
    # step 3 synthesis of algorithms with C++ code and generate lambdas (guarantees)