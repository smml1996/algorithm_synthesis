from typing import List, Dict, Tuple
import numpy as np
import sys, os


sys.path.append(os.getcwd()+"/..")
from utils import Precision, find_enum_object
from pomdp import POMDPAction, POMDPVertex, default_guard
from qpu_utils import BasisGates, Op
from ibm_noise_models import HardwareSpec, Instruction, NoiseModel, load_config_file
import qmemory
from qstates import QuantumState
from cmemory import ClassicalState
from experiments_utils import YGateExperimentId, actions_sequence_to_algorithm_node, check_files, generate_algs_vs_file, generate_configs, generate_diff_algorithms_file, generate_embeddings, generate_mc_guarantees_file, generate_pomdps, get_allowed_hardware, get_config_path, get_custom_guarantee, get_num_qubits_to_hardware, get_pomdp_path, get_project_settings

class YGateInstance:
    def __init__(self, embedding, experiment_id: YGateExperimentId):
        assert isinstance(experiment_id, YGateExperimentId)
        self.embedding = embedding
        self.experiment_id = experiment_id
        self.initial_distribution = None
        
        assert experiment_id in [YGateExperimentId.MAIN99, YGateExperimentId.MAIN999, YGateExperimentId.MAIN9999, YGateExperimentId.MAIN_THERM99, YGateExperimentId.MAIN_THERM999, YGateExperimentId.MAIN_THERM9999]
        assert len(embedding) == 2
        self.qubits_used = self.embedding.values()
        self.get_initial_distribution()
        
        # precompute target state
        H0 = Instruction(self.embedding[0], Op.H).get_gate_data()
        CX01 = Instruction(self.embedding[1], Op.CNOT, self.embedding[0]).get_gate_data()
        Y1 = Instruction(self.embedding[1], Op.Y).get_gate_data()
        target_state = QuantumState(0, qubits_used=self.embedding.values())
        
        target_state = qmemory.handle_write(target_state, H0)
        target_state = qmemory.handle_write(target_state, CX01)
        target_state = qmemory.handle_write(target_state, Y1)
        self.target_state = target_state
        
    
    def get_initial_distribution(self):
        
        H0 = Instruction(self.embedding[0], Op.H).get_gate_data()
        CX01 = Instruction(self.embedding[1], Op.CNOT, self.embedding[0]).get_gate_data()
        
        self.initial_distribution = []
        
        initial_cs = ClassicalState()
        
        initial_qs = QuantumState(0, qubits_used=self.embedding.values())
        
        # assuming input qubit is at address 0 and output qubit is at address 1
        ## prepare bell state
        initial_qs = qmemory.handle_write(initial_qs, H0)
        initial_qs = qmemory.handle_write(initial_qs, CX01)
        
        self.initial_distribution.append(((initial_qs, initial_cs), 1.0))
        
    def get_reward(self, vertex: POMDPVertex) -> float:
        return int(vertex.quantum_state == self.target_state)
    
def get_experiments_actions(noise_model: NoiseModel, embedding: dict[int, int], experiment_id: YGateExperimentId):
    assert len(embedding.keys()) == 2
    
    assert experiment_id in [YGateExperimentId.MAIN99,YGateExperimentId.MAIN999, YGateExperimentId.MAIN9999, YGateExperimentId.MAIN_THERM99, YGateExperimentId.MAIN_THERM999, YGateExperimentId.MAIN_THERM9999]
    result = []
    if noise_model.basis_gates in [BasisGates.TYPE2, BasisGates.TYPE4, BasisGates.TYPE9, BasisGates.TYPE10, BasisGates.TYPE11]:
        sx_instruction = [Instruction(embedding[1], Op.SX)]
        result.append(POMDPAction("SX", sx_instruction))
        
        rzpi_instruction = [Instruction(embedding[1], Op.RZ, params=[2*np.pi])]
        result.append(POMDPAction("RZPI", rzpi_instruction))
        
        rz3pi_instruction = [Instruction(embedding[1], Op.RZ, params=[3*np.pi])]
        result.append(POMDPAction("RZ3PI", rz3pi_instruction))
        
        rzpihalf_instruction = [Instruction(embedding[1], Op.RZ, params=[np.pi/2])]
        result.append(POMDPAction("RZPIHalf", rzpihalf_instruction))
        
        s_instruction = [Instruction(embedding[1], Op.RZ, params=[-np.pi/2])]
        result.append(POMDPAction("S", s_instruction))
        
        x_instruction = [Instruction(embedding[1], Op.X)]
        result.append(POMDPAction("X", x_instruction))
    else:
        assert noise_model.basis_gates == BasisGates.TYPE8
        
        u3_instruction = [Instruction(embedding[1], Op.U3, params=[np.pi, np.pi/2, np.pi/2])]
        result.append(POMDPAction("U3", u3_instruction))
        
        h_instruction = Instruction(embedding[1], Op.H).to_basis_gate_impl(noise_model.basis_gates)
        result.append(POMDPAction("H", h_instruction))
        
        s_instruction = Instruction(embedding[1], Op.S).to_basis_gate_impl(noise_model.basis_gates)
        result.append(POMDPAction("S", s_instruction))
        
        z_instruction = Instruction(embedding[1], Op.Z).to_basis_gate_impl(noise_model.basis_gates)
        result.append(POMDPAction("Z", z_instruction))
        
        x_instruction = Instruction(embedding[1], Op.X).to_basis_gate_impl(noise_model.basis_gates)
        result.append(POMDPAction("X", x_instruction))
        
        ry_instruction = Instruction(embedding[1], Op.RY, params=[np.pi]).to_basis_gate_impl(noise_model.basis_gates)
        result.append(POMDPAction("RY", ry_instruction))
        
    return result

def get_thermalization_setup(experiment_id: YGateExperimentId) -> bool:
    if experiment_id in [YGateExperimentId.MAIN99, YGateExperimentId.MAIN999, YGateExperimentId.MAIN9999]:
        return False
    elif experiment_id in [YGateExperimentId.MAIN_THERM99,YGateExperimentId.MAIN_THERM999,YGateExperimentId.MAIN_THERM9999]:
        return True
    else:
        raise Exception(f"thermalization setup not specified for experiment {experiment_id}")


def get_hardware_scenarios(hardware_spec: HardwareSpec, experiment_id) -> List[Dict[int, int]]:
    assert experiment_id in [YGateExperimentId.MAIN99, YGateExperimentId.MAIN999, YGateExperimentId.MAIN9999, YGateExperimentId.MAIN_THERM99, YGateExperimentId.MAIN_THERM999, YGateExperimentId.MAIN_THERM9999]
    with_thermalization = get_thermalization_setup(experiment_id)
    noise_model = NoiseModel(hardware_spec, thermal_relaxation=with_thermalization)
    
    
    if noise_model.basis_gates in [BasisGates.TYPE2, BasisGates.TYPE4, BasisGates.TYPE9, BasisGates.TYPE10, BasisGates.TYPE11]:
        ops_used = [
            Op.SX, Op.RZ, Op.X
        ]
    else:
        assert noise_model.basis_gates == BasisGates.TYPE8
        ops_used = [
            Op.U3,
            Op.U2,
            Op.U1
        ]
    qubits = set()
    for op in ops_used:
        most_noisy = noise_model.get_most_noisy_qubit(op, top=1)
        qubits.add(most_noisy[0][1])
    
    answer = []
    for qubit in qubits:
        if qubit == 0:
            qubit0 = 1
        else:
            qubit0 = 0
        answer.append({0: qubit0, 1: qubit})
    return answer

def get_guard(experiment_id):
    return default_guard

def set_precision(experiment_id):
    Precision.PRECISION = 8
    Precision.update_threshold()
    
def get_min_max_horizon(experiment_id) -> Tuple[int, int]:
    return 2, 6


def get_target_precision(experiment_id: YGateExperimentId) -> float:
    if experiment_id in [YGateExperimentId.MAIN99, YGateExperimentId.MAIN_THERM99]:
        return 0.99
    if experiment_id in [YGateExperimentId.MAIN999, YGateExperimentId.MAIN_THERM999]:
        return 0.999
    if experiment_id in [YGateExperimentId.MAIN9999, YGateExperimentId.MAIN_THERM9999]:
        return 0.9999
    
def check_specific_algorithms_accuracy(batches): 
    action_sequences = [[
        [Instruction(1, Op.Y)],
        [Instruction(1, Op.U3, params=[np.pi, np.pi/2, np.pi/2])],
        [
            Instruction(1, Op.H), 
            Instruction(1, Op.S), 
            Instruction(1, Op.S),
            Instruction(1, Op.H),
            Instruction(1, Op.S), 
            Instruction(1, Op.S)
        ],
        [
            Instruction(1, Op.S), 
            Instruction(1, Op.S),
            Instruction(1, Op.H), 
            Instruction(1, Op.S), 
            Instruction(1, Op.S),
            Instruction(1, Op.H),
        ],
        [
            Instruction(1, Op.Z),
            Instruction(1, Op.X)
        ],
        [Instruction(1, Op.RY, params=[np.pi])]
    ]]
    
    specific_algorithms = [actions_sequence_to_algorithm_node(x) for x in action_sequences]
    
    # start testing
    file = open("y_gate_specific_algs.csv", "w")
    columns = [
        "experiment_id",
        "hardware_spec",
        "embedding_index",
        "algorithm",
        "guarantee"
    ]
    file.write(",".join(columns) + "\n")
    
    for experiment_id in YGateExperimentId:
        for (num_qubits, hardware_specs) in batches.keys():
            config_path = get_config_path(experiment_id, num_qubits)
            config = load_config_file(config_path, type(experiment_id))
            for hardware_spec in hardware_specs:
                embeddings = get_hardware_scenarios(hardware_spec, experiment_id)
                for (embedding_index, embedding) in enumerate(embeddings):
                    pomdp_path = get_pomdp_path(config, hardware_spec, embedding_index)
                    for (algorithm_index, algorithm_node) in enumerate(specific_algorithms):
                        guarantee = get_custom_guarantee(algorithm_node, pomdp_path, config)
                        columns = [
                            experiment_id.value,
                            hardware_spec.value,
                            str(embedding_index),
                            str(algorithm_index),
                            str(guarantee)
                        ]
                        file.write(",".join(columns) + "\n")
                        
    file.close()
            
    
if __name__ == "__main__":
    settings = get_project_settings()
    project_path = settings["PROJECT_PATH"]
    
    experiment_name = sys.argv[1]
    process_name = sys.argv[2]
    batch_name = None
    if len(sys.argv) > 3:
        batch_name = sys.argv[3]
    
    experiment_id = find_enum_object(experiment_name, YGateExperimentId)
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
            generate_embeddings(experiment_id, num_qubits, get_hardware_embeddings=get_hardware_scenarios)
    elif process_name == "gen_pomdps":
        # generate POMDPS
        if batch_name is None:
            for num_qubits in batches.keys():
                config_path = get_config_path(experiment_id, num_qubits)
                generate_pomdps(experiment_id, num_qubits, get_experiments_actions, YGateInstance, guard=get_guard(experiment_id), set_hidden_index=True, WITH_THERMALIZATION=with_thermalization)
        else:
            config_path = get_config_path(experiment_id, batch_name)
            generate_pomdps(experiment_id, batch_name, get_experiments_actions, YGateInstance, guard=get_guard(experiment_id), set_hidden_index=False, WITH_THERMALIZATION=with_thermalization)
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
    elif process_name == "check_specific_algs":
        check_specific_algorithms_accuracy(batches)
    else:
        raise Exception("Invalid process name", process_name)
        