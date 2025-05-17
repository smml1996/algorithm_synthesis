from copy import deepcopy
import sys, os
from typing import Dict, Tuple, List
sys.path.append(os.getcwd() + "/..")

from problems.ghz import are_adjacent_qubits, get_valid_third, is_repeated_embedding
from utils import Precision, find_enum_object
from pomdp import POMDPAction, POMDPVertex, default_guard
from qpu_utils import BasisGates, Op
from ibm_noise_models import HardwareSpec, Instruction, NoiseModel
import qmemory
from qstates import QuantumState
from cmemory import ClassicalState
from experiments_utils import SwapExperimentId, check_files, generate_algs_vs_file, generate_configs, generate_diff_algorithms_file, generate_embeddings, generate_mc_guarantees_file, generate_pomdps, get_allowed_hardware, get_config_path, get_num_qubits_to_hardware, get_project_settings

class SwapInstance:
    def __init__(self, embedding: Dict[int, int], experiment_id: SwapExperimentId):
        assert isinstance(experiment_id, SwapExperimentId)
        self.embedding = embedding
        self.experiment_id = experiment_id
        self.initial_distribution = None
        
        assert experiment_id in [
            SwapExperimentId.SWAP90, 
            SwapExperimentId.SWAP95, 
            SwapExperimentId.SWAP98, 
            SwapExperimentId.SWAP99
        ]
        
        assert len(embedding) == 6
        self.qubits_used = self.embedding.values()
        self.get_initial_distribution()
        
        # precompute target state
        # 0 | 1 | 2 || 3 | 4 | 5
        self.qubit0 = embedding[3]
        self.qubit1 = embedding[4]
        gates = [
            Instruction(self.qubit0, Op.SWAP, self.qubit1).get_gate_data(),
        ]
        self.target_state = qmemory.get_linear_op_state(gates, embedding, 3)
        
    def get_initial_distribution(self):
        self.initial_distribution = []
        initial_cs = ClassicalState()
        initial_qs = qmemory.get_linear_op_state([], self.embedding, 3)
        
        self.initial_distribution.append(((initial_qs, initial_cs), 1.0))
        
    def get_reward(self, vertex: POMDPVertex) -> float:
        return int(vertex.quantum_state == self.target_state)

def append_swap_or_cx(instruction1, instruction2, noise_model, index1, index2, result):
    if instruction1 in noise_model.instructions_to_channel.keys():
        if instruction2 in noise_model.instructions_to_channel.keys():
            result.append(POMDPAction(f"SWAP{index1}{index2}", [
                instruction1,
                instruction2,
                instruction1
            ]))
            result.append(POMDPAction(f"SWAP{index2}{index1}", [
                instruction2,
                instruction1,
                instruction2
            ]))
        else:
            result.append(POMDPAction(f"CX{index1}{index2}", [
                instruction1
            ]))
    elif instruction2 in noise_model.instructions_to_channel.keys():
        result.append(POMDPAction(f"CX{index2}{index1}", [
            instruction2
        ]))

def get_experiments_actions(noise_model: NoiseModel, embedding: dict[int, int], experiment_id: SwapExperimentId):
    assert experiment_id in [
            SwapExperimentId.SWAP90, 
            SwapExperimentId.SWAP95, 
            SwapExperimentId.SWAP98, 
            SwapExperimentId.SWAP99
        ]
    result = []
    assert len(embedding) == 6
    
    qubit0 = embedding[3]
    qubit1 = embedding[4]
    qubit2 = embedding[5]
    
    instruction_cx01 = Instruction(qubit1, Op.CNOT, control=qubit0)
    instruction_cx10 = Instruction(qubit0, Op.CNOT, control=qubit1)
    append_swap_or_cx(instruction_cx01, instruction_cx10, noise_model, 0, 1, result)

    instruction_cx12 = Instruction(qubit2, Op.CNOT, control=qubit1)
    instruction_cx21 = Instruction(qubit1, Op.CNOT, control=qubit2)
    append_swap_or_cx(instruction_cx12, instruction_cx21, noise_model, 1, 2, result)
    
    instruction_cx02 = Instruction(qubit2, Op.CNOT, control=qubit0)
    instruction_cx20 = Instruction(qubit0, Op.CNOT, control=qubit2)
    append_swap_or_cx(instruction_cx02, instruction_cx20, noise_model, 0, 2, result)
    
    return result
    
    

def get_thermalization_setup(experiment_id: SwapExperimentId) ->bool:
    return False

def get_unused_qubit(used_values, num_qubits) -> int:
    for new_index in range(num_qubits):
        if new_index not in used_values:
            return new_index
    return None

def get_hardware_scenarios(hardware_spec: HardwareSpec, experiment_id: SwapExperimentId) -> List[Dict[int, int]]:
    assert experiment_id in [
        SwapExperimentId.SWAP90, 
        SwapExperimentId.SWAP95, 
        SwapExperimentId.SWAP98, 
        SwapExperimentId.SWAP99
    ]
    
    with_thermalization = get_thermalization_setup(experiment_id)
    noise_model = NoiseModel(hardware_spec, thermal_relaxation=with_thermalization)
    
    couplers = noise_model.get_most_noisy_couplers()
    result = []
    for (coupler, prob_) in couplers:
        if len(result) == 3:
            break
        third_qubit = get_valid_third(noise_model, coupler)
        if third_qubit is not None:
            d_temp = dict()
            d_temp[3] = coupler[0]
            d_temp[4] = coupler[1]
            d_temp[5] = third_qubit
            d_temp[0] = get_unused_qubit(d_temp.values(), noise_model.num_qubits)
            d_temp[1] = get_unused_qubit(d_temp.values(), noise_model.num_qubits)
            d_temp[2] = get_unused_qubit(d_temp.values(), noise_model.num_qubits)
            assert third_qubit != coupler[0]
            assert third_qubit != coupler[1]
            assert coupler[0] != coupler[1]
            
            contains_none = False
            for q in d_temp.values():
                if q is None:
                    contains_none = True
            if not contains_none:
                if not is_repeated_embedding(result, d_temp):
                    result.append(deepcopy(d_temp))  
    return result

def get_guard(experiment_id: SwapExperimentId):
    return default_guard

def set_precision(experiment_id: SwapExperimentId):
    Precision.PRECISION = 8
    Precision.update_threshold()
    
def get_min_max_horizon(experiment_id: SwapExperimentId) -> Tuple[int, int]:
    return 1, 3

def get_target_precision(experiment_id: SwapExperimentId) -> float:
    if experiment_id == SwapExperimentId.SWAP90:
        return 0.9
    if experiment_id == SwapExperimentId.SWAP95:
        return 0.95
    if experiment_id == SwapExperimentId.SWAP98:
        return 0.98
    if experiment_id == SwapExperimentId.SWAP99:
        return 0.99
    raise Exception("precision not set for experiment", experiment_id)


if __name__ == "__main__":
    settings = get_project_settings()
    project_path = settings["PROJECT_PATH"]
    
    experiment_name = sys.argv[1]
    process_name = sys.argv[2]
    batch_name = None
    if len(sys.argv) > 3:
        batch_name = sys.argv[3]
    
    experiment_id = find_enum_object(experiment_name, SwapExperimentId)
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
        generate_configs(experiment_id, min_horizon=min_horizon, max_horizon=max_horizon, allowed_hardware=allowed_hardware, opt_technique="target", reps=get_target_precision(experiment_id))
    
        print("generating embedding files...")
        for num_qubits in batches.keys():
            config_path = get_config_path(experiment_id, num_qubits)
            generate_embeddings(experiment_id, num_qubits, get_hardware_embeddings=get_hardware_scenarios)
    elif process_name == "gen_pomdps":
        # generate POMDPS
        if batch_name is None:
            for num_qubits in batches.keys():
                config_path = get_config_path(experiment_id, num_qubits)
                generate_pomdps(experiment_id, num_qubits, get_experiments_actions, SwapInstance, guard=get_guard(experiment_id), set_hidden_index=True, WITH_THERMALIZATION=with_thermalization)
        else:
            config_path = get_config_path(experiment_id, batch_name)
            generate_pomdps(experiment_id, batch_name, get_experiments_actions, SwapInstance, guard=get_guard(experiment_id), set_hidden_index=False, WITH_THERMALIZATION=with_thermalization)
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
    