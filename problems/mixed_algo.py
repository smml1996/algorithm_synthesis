from copy import deepcopy
from enum import Enum
from math import pi
import os, sys
from typing import Dict, List, Tuple

import numpy as np
sys.path.append(os.getcwd()+"/..")

from utils import are_matrices_equal, Precision, find_enum_object
from ibm_noise_models import HardwareSpec, Instruction, MeasChannel, NoiseModel
from qstates import QuantumState
from qpu_utils import Op
from cmemory import ClassicalState, cread
from pomdp import POMDPAction, POMDPVertex, default_guard
import qmemory
from experiments_utils import ReadoutNoise, rho_qubit0, rho_qubit1, run_bellmaneq
from experiments_utils import check_files, generate_algs_vs_file, generate_configs, generate_diff_algorithms_file, generate_embeddings, generate_mc_guarantees_file, generate_pomdps, get_config_path, get_embeddings_path, get_num_qubits_to_hardware, get_project_settings, bell_state_pts, load_embeddings

MAX_PRECISION = 10
WITH_THERMALIZATION = False

class MixedAlgs(Enum):
    toy = "toy"
    @property
    def exp_name(self):
        return "mixed"

class MixedAlgsInstance:
    def __init__(self, embedding, experiment_id: MixedAlgs):
        """_summary_

        Args:
            num_qubits (int): _description_
            instruction_set (List[Instruction]): _description_
            embedding (Dict[int, int]): a mapping from logical qubits to physical qubits
        """
        assert isinstance(experiment_id, MixedAlgs)
        self.embedding = embedding
        self.experiment_id = experiment_id
        self.initial_distribution = None
        # check embedding
        assert 0 in self.embedding.keys()
        assert len(self.embedding.keys()) == 1
        self.qubits_used = [self.embedding[0]]
        self.get_initial_distribution()
      
    def get_reward(self, vertex: POMDPVertex) -> float:     
        return int(cread(vertex.classical_state, 0) == vertex.hidden_index)
    
    def get_condition(self, vertex: POMDPVertex) -> str:
        if self.get_reward(vertex) == 1:
            return f"{vertex.id} >= 0.6"
        return None

    def  get_initial_distribution(self):
        self.initial_distribution = []
        initial_cs = ClassicalState()
        
        H0 = Instruction(self.embedding[0], Op.H).get_gate_data()
        
        # append |0> state
        zero = QuantumState(0, qubits_used=self.embedding.values())
        self.initial_distribution.append(((zero, initial_cs), 0.5))
        
        # append |+>
        plus = qmemory.handle_write(zero, H0)
        self.initial_distribution.append(((plus, initial_cs), 0.5))
    
def get_experiments_actions(noise_model, embedding, experiment_id):
    assert isinstance(noise_model, NoiseModel)
    assert isinstance(experiment_id, MixedAlgs)
    
    actions = []

    if experiment_id == MixedAlgs.toy:    
            actions.append(
                POMDPAction(name="H0",
                            instruction_sequence=Instruction(embedding[0], Op.H).to_basis_gate_impl(noise_model.basis_gates))
            )
            
            actions.append(
                POMDPAction(name="MEAS-0", 
                            instruction_sequence=[Instruction(embedding[0], Op.MEAS, real_target=0)])
            )
            
            DETERMINE0 = POMDPAction("WRITE0-0", [Instruction(0, Op.WRITE0)])
            actions.append(DETERMINE0)

            DETERMINEPlus = POMDPAction("WRITE1-0", [Instruction(0, Op.WRITE1)])
            actions.append(DETERMINEPlus)
    else:
        raise Exception(f"No actions specified for experiment: {experiment_id}")
    return actions

def get_unused_qubit(noise_model: NoiseModel, used_qubits: List[int]) -> int:
    for q in range(noise_model.num_qubits):
        if q not in used_qubits:
            return q
    raise Exception(f"failed to find unused qubit with used={used_qubits} and hardware_spec={noise_model.hardware_spec}")

def get_pivot_qubits(noise_model: NoiseModel, min_indegree=0):
    result = set()
    noises = []
    if noise_model.hardware_spec == HardwareSpec.MELBOURNE:
        noise_model.num_qubits = 14
    for qubit in range(noise_model.num_qubits):
        if noise_model.get_qubit_indegree(qubit) >= min_indegree:
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
        result.add(temp[len(temp)-1].target)

    temp = sorted(noises, key=lambda x: x.abs_diff)
    if temp[0].abs_diff != temp[len(temp)-1].abs_diff:
        result.add(temp[0].target)
        assert (temp[0].abs_diff < temp[len(temp)-1].abs_diff)
    return result

def get_hardware_scenarios(hardware_spec: HardwareSpec, experiment_id) -> List[Dict[int, int]]:
    ''' returns hardware scenarios (embeddings) for a given hardware specification
    '''
    noise_model = NoiseModel(hardware_spec, thermal_relaxation=False)
    answer = []
    pivot_qubits = get_pivot_qubits(noise_model)
    if experiment_id in [MixedAlgs.toy]:
       for i in pivot_qubits:
           embedding = dict()
           embedding[0] = i
           answer.append(deepcopy(embedding))
    else:
        raise Exception(f"get_hardware_scenarios for experiment {experiment_id} not implemented")
    return answer

def get_allowed_hardware(experiment_id: MixedAlgs, with_thermalization: bool) :
    return HardwareSpec

def get_thermalization_setup(experiment_id):
    return False

def set_precision(experiment_id):
    Precision.PRECISION = 5
    Precision.update_threshold()
    
def get_min_max_horizon(experiment_id) -> Tuple[int, int]:
    return 1, 3

def get_guard(experiment_id):
    return default_guard

if __name__ == "__main__":
    settings = get_project_settings()
    project_path = settings["PROJECT_PATH"]
    
    experiment_name = sys.argv[1]
    process_name = sys.argv[2]
    batch_name = None
    if len(sys.argv) > 3:
        batch_name = sys.argv[3]
    
    experiment_id = find_enum_object(experiment_name, MixedAlgs)
    with_thermalization = get_thermalization_setup(experiment_id)
    if experiment_id is None:
        raise Exception("Experiment name:", experiment_name, " does not match any element of the enum")
    
    set_precision(experiment_id)
    allowed_hardware = get_allowed_hardware(experiment_id, with_thermalization)
    batches = get_num_qubits_to_hardware(with_thermalization, allowed_hardware)
    
    if process_name == "setup":
        min_horizon, max_horizon = get_min_max_horizon(experiment_id)
        # generate configuration files
        print("Generating configuration files...")
        generate_configs(experiment_id, min_horizon=min_horizon, max_horizon=max_horizon, allowed_hardware=allowed_hardware)
    
        print("generating embedding files...")
        for num_qubits in batches.keys():
            config_path = get_config_path(experiment_id, num_qubits)
            generate_embeddings(experiment_id, num_qubits, get_hardware_embeddings=get_hardware_scenarios)
    elif process_name == "gen_pomdps":
        # generate POMDPS
        if batch_name is None:
            for num_qubits in batches.keys():
                config_path = get_config_path(experiment_id, num_qubits)
                generate_pomdps(experiment_id, num_qubits, get_experiments_actions, MixedAlgsInstance, guard=get_guard(experiment_id), set_hidden_index=True, WITH_THERMALIZATION=with_thermalization, optimize_graph=False)
        else:
            config_path = get_config_path(experiment_id, batch_name)
            generate_pomdps(experiment_id, batch_name, get_experiments_actions, MixedAlgsInstance, guard=get_guard(experiment_id), set_hidden_index=True, WITH_THERMALIZATION=with_thermalization, optimize_graph=False)
    elif process_name == "mc_guarantees":
        # markov chain guarantees for each algorithm
        generate_mc_guarantees_file(experiment_id, allowed_hardware, get_hardware_scenarios, get_experiments_actions, WITH_THERMALIZATION=with_thermalization)
    elif process_name == "diff_algs_file":
        generate_diff_algorithms_file(experiment_id, allowed_hardware, get_hardware_scenarios, get_experiments_actions, with_thermalization=with_thermalization)
    elif process_name == "check_files":
        # checks all algorithms for all hardware scenarios have been synthesized and we have all lambdas
        check_files(experiment_id, allowed_hardware, with_thermalization=with_thermalization)
    elif process_name == "diffs_algs_vs":
        # compare performance of all all algorithms in the diffs file
        generate_algs_vs_file(experiment_id, allowed_hardware, get_hardware_scenarios, get_experiments_actions, with_thermalization=with_thermalization)
    else:
        raise Exception("Invalid process name", process_name)
 
        