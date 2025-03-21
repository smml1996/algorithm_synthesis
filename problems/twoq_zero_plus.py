from copy import deepcopy
from enum import Enum
from math import pi
import os, sys
from typing import Dict, List
sys.path.append(os.getcwd()+"/..")

from utils import are_matrices_equal, Precision
from ibm_noise_models import HardwareSpec, Instruction, NoiseModel
from qstates import QuantumState
from qpu_utils import Op
from cmemory import ClassicalState, cread
from pomdp import POMDPAction, POMDPVertex
import qmemory
from experiments_utils import generate_configs, generate_embeddings, generate_pomdps, get_config_path, get_num_qubits_to_hardware, get_project_settings
from zero_plus import get_pivot_qubits

MAX_PRECISION = 5
WITH_THERMALIZATION = False

class TwoQZeroPlusExperimentID(Enum):
    TWOQ = "twoq"
    @property
    def exp_name(self):
        return "twoqzeroplus"
    
class ZeroPlusInstance:
    def __init__(self, embedding, experiment_id: TwoQZeroPlusExperimentID):
        """_summary_

        Args:
            num_qubits (int): _description_
            instruction_set (List[Instruction]): _description_
            embedding (Dict[int, int]): a mapping from logical qubits to physical qubits
        """
        assert isinstance(experiment_id, TwoQZeroPlusExperimentID)
        self.embedding = embedding
        self.experiment_id = experiment_id
        self.initial_distribution = None
        self.qubits_used = [self.embedding[0], self.embedding[1]]
        # check embedding
        
        if self.experiment_id in [TwoQZeroPlusExperimentID.TWOQ]:
            assert len(self.embedding.keys()) == 2 # 2 + 2 qubit for hidden indices
        else:
            raise Exception(f"Setup not set for experiment {self.experiment_id}")
        self.get_initial_distribution()
        
    def get_initial_distribution(self):
        self.initial_distribution = []
        initial_cs = ClassicalState()
        
        H0 = Instruction(self.embedding[0], Op.H).get_gate_data()
        
        # append |0> state
        zero = QuantumState(0, qubits_used=self.embedding.values())
        self.initial_distribution.append(((zero, initial_cs), 0.5))
        
        # append |+>
        plus = qmemory.handle_write(zero, H0)
        self.initial_distribution.append(((plus, initial_cs), 0.5))
    
    def get_reward(self, vertex: POMDPVertex) -> float:     
        return cread(vertex.classical_state, 0) == vertex.hidden_index
    
def get_experiments_actions(noise_model, embedding, experiment_id):
    assert isinstance(noise_model, NoiseModel)
    assert isinstance(experiment_id, TwoQZeroPlusExperimentID)
    
    actions = []
    if experiment_id == TwoQZeroPlusExperimentID.TWOQ:
        assert len(embedding) == 2
        h0_instruction = Instruction(embedding[0], Op.H).to_basis_gate_impl(noise_model.basis_gates)
        h1_instruction = Instruction(embedding[1], Op.H).to_basis_gate_impl(noise_model.basis_gates)
        
        h0_action = POMDPAction("H0", h0_instruction)
        actions.append(h0_action)
        
        h1_action = POMDPAction("H1", h1_instruction)
        actions.append(h1_action)
        
        ry0_instruction = Instruction(embedding[0], Op.RY, params=[pi/4]).to_basis_gate_impl(noise_model.basis_gates)
        ry0_action = POMDPAction("RY0", ry0_instruction)
        actions.append(ry0_action)
        
        ry1_instruction =  Instruction(embedding[1], Op.RY, params=[pi/4]).to_basis_gate_impl(noise_model.basis_gates)
        ry1_action = POMDPAction("RY1", ry1_instruction)
        actions.append(ry1_action)
        
        rycx01_action = POMDPAction("rycx01", ry1_instruction + [Instruction(embedding[1], Op.CNOT, control=embedding[0]), Instruction(4, Op.WRITE1)])
        actions.append(rycx01_action)
        
        hcx10_action = POMDPAction("hcx10", h1_instruction + [Instruction(embedding[0], Op.CNOT, control=embedding[1]), Instruction(4, Op.WRITE1)])
        actions.append(hcx10_action)
        
        hcu10_action = POMDPAction("hcu10", [Instruction(4, Op.WRITE1)] + h1_instruction + 
                        # control-ry gate
                        Instruction(embedding[0], Op.CNOT, control=embedding[1]).to_basis_gate_impl(noise_model.basis_gates)+
                        Instruction(embedding[0], Op.RY, params=[-pi/4]).to_basis_gate_impl(noise_model.basis_gates) +
                        Instruction(embedding[0], Op.CNOT, control=embedding[1]).to_basis_gate_impl(noise_model.basis_gates)+
                        ry0_instruction
                        )
        actions.append(hcu10_action)
        
        meas_action = POMDPAction("MEAS", [
            Instruction(embedding[0], Op.MEAS, real_target=0),
            Instruction(embedding[1], Op.MEAS, real_target=1),
            Instruction(2, Op.WRITE1)
        ])
        actions.append(meas_action)
        
        DETERMINE0 = POMDPAction("IS0", [Instruction(0, Op.WRITE0), Instruction(3, Op.WRITE1)])
        DETERMINEPlus = POMDPAction("ISPlus", [Instruction(0, Op.WRITE1), Instruction(3, Op.WRITE1)])
        actions.append(DETERMINE0)
        actions.append(DETERMINEPlus)
        
    return actions

def get_hardware_scenarios(hardware_spec: HardwareSpec, experiment_id) -> List[Dict[int, int]]:
    ''' returns hardware scenarios (embeddings) for a given hardware specification
    '''
    noise_model = NoiseModel(hardware_spec, thermal_relaxation=False)
    answer = []
    pivot_qubits = get_pivot_qubits(noise_model)
    if experiment_id in [TwoQZeroPlusExperimentID.TWOQ]:
        most_noisy_coupler = noise_model.get_most_noisy_couplers()[0][0]
        embedding = dict()
        control = most_noisy_coupler[0]
        target = most_noisy_coupler[1]
        embedding[0] = target
        embedding[1] = control
        answer.append(deepcopy(embedding))
    else:
        raise Exception(f"get_hardware_scenarios for experiment {experiment_id} not implemented")
    return answer

def twoq_guard(vertex: POMDPVertex, embedding: Dict[int, int], action: POMDPAction) -> bool:
    classical_state = vertex.classical_state
    if cread(classical_state, 3) == 1:
        return False
    
    if cread(classical_state, 2) == 1:
        return action.name in ["IS0", "ISPlus"]
    
    if cread(classical_state, 4) == 1:
        return not (action.name in ["hcu10", "hcx10", "rycx01"])
    
    return True

def get_allowed_hardware():
    ''' We will only run experiments on quantum hardware that has CNOT gates in its basis gate set
    '''
    allowed_harware = []
    for hardware_spec in HardwareSpec:
        noise_model = NoiseModel(hardware_spec, thermal_relaxation=WITH_THERMALIZATION)
        if Op.CNOT in noise_model.basis_gates.value:
            allowed_harware.append(hardware_spec)
    return allowed_harware

if __name__ == "__main__":
    allowed_hardware = get_allowed_hardware()
    
    experiment_id = TwoQZeroPlusExperimentID.TWOQ
    Precision.PRECISION = MAX_PRECISION
    Precision.update_threshold()
    settings = get_project_settings()
    project_path = settings["PROJECT_PATH"]
    
    # print("Generating configuration files...")
    # generate_configs(experiment_id, min_horizon=3, max_horizon=5)
    
    batches = get_num_qubits_to_hardware(WITH_THERMALIZATION, allowed_hardware)
    
    # print("generating embedding files...")
    # for num_qubits in batches.keys():
    #     config_path = get_config_path(experiment_id, num_qubits)
    #     generate_embeddings(experiment_id, num_qubits, get_hardware_embeddings=get_hardware_scenarios)
        
    for num_qubits in batches.keys():
        config_path = get_config_path(experiment_id, num_qubits)
        generate_pomdps(experiment_id, num_qubits, get_experiments_actions, ZeroPlusInstance, guard=twoq_guard)
    
    
    
    
        