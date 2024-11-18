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
from math import pi   
from enum import Enum
from experiments_utils import ReadoutNoise, default_load_embeddings, directory_exists, generate_configs, generate_embeddings, get_config_path, get_configs_path, get_embeddings_path, get_num_qubits_to_hardware, get_project_path, get_project_settings

from ghz import GHZExperimentID, GHZInstance

WITH_TERMALIZATION = False
MAX_PRECISION = 10

class ThreeQCode(Enum):
    EXP1 = "exp1"
    
class ThreeQInstance:
    def __init__(self, embedding) -> None:
        '''
        In embedding assume the GHZ state is in qubits 0-2, qubit 3 is for Z-basis meas. and qubit 4 is for X-basis measurement
        '''
        self.embedidng = embedding
        self.initial_state = None
        self.ghz_instance = GHZInstance(embedding)
        self.target_state = None
        self.get_initial_states()
        self.set_target_state()
        
        for i in range(5):
            assert i in self.embedding.keys()
            
    def get_initial_states(self):
        self.initial_state = self.ghz_instance.initial_state
    
    def set_target_state(self):
        self.target_state = self.ghz_instance.target_state.get_density_matrix()
        
    def get_reward(self, hybrid_state) -> float:
        qs, _ = hybrid_state
        assert isinstance(qs, QuantumState)
        current_rho = qs.multi_partial_trace(remove_indices=[self.embedding[3], self.embeddding[4]])
        
        if are_matrices_equal(current_rho, self.target_state):
            return 1
        else:
            return 0


if __name__ == "__main__":
    
    # set up working directory
    project_path = get_project_path()
    output_dir = os.path.join(project_path, "results", "3q_code")
    
    # init results csv
    output_file_path = os.path.join(output_dir, "results.csv")
    output_file = open(output_file_path, "w")
    columns = ["embedding_id",
               "my_ghz",
               "default_ghz",
                "my_bitflip",
               "default_bitflip",
               "my_phaseflip",
               "default_phaseflip",
               "my_swap_Z",
               "default_swap_Z",
               "my_swap_X",
               "default_swap_X",
               "my_acc",
               "default_acc",
               "diff"
               ]
    output_file.write(",".join(columns) + "\n")
    
    
    # rows computations begins here
    for hardware_spec in HardwareSpec:
        noise_model = NoiseModel(hardware_spec, thermal_relaxation=WITH_TERMALIZATION)
        
        ghz_embeddings = 1
        z_pivot_qubits = 1
        x_pivot_qubits = 1
    
    
    