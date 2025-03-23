from copy import deepcopy
from enum import Enum
from math import pi
import os, sys
from typing import Dict, List, Tuple
sys.path.append(os.getcwd()+"/..")

from utils import Precision
from ibm_noise_models import HardwareSpec, Instruction, NoiseModel
from qstates import QuantumState
from qpu_utils import BasisGates, Op
from cmemory import ClassicalState, cread
from pomdp import POMDPAction, POMDPVertex
import qmemory
from experiments_utils import generate_configs, generate_embeddings, generate_pomdps, get_config_path, get_num_qubits_to_hardware, get_project_settings
from bitflip import get_pivot_qubits

MAX_PRECISION = 5
WITH_THERMALIZATION = False

class TwoQZeroPlusExperimentID(Enum):
    TWOQ = "twoq"
    TWOQParity = "twoqparity"
    TWOQ2 = "twoq2"
    
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
        assert self.experiment_id in [TwoQZeroPlusExperimentID.TWOQ, TwoQZeroPlusExperimentID.TWOQ2]
        assert len(self.embedding.keys()) == 2 # 2 + 2 qubit for hidden indices
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
        return int(cread(vertex.classical_state, 0) == vertex.hidden_index)
    
def get_experiments_actions(noise_model: NoiseModel, embedding, experiment_id):
    assert isinstance(noise_model, NoiseModel)
    assert isinstance(experiment_id, TwoQZeroPlusExperimentID)
    
    # some instructions
    h0_instruction = Instruction(embedding[0], Op.H).to_basis_gate_impl(noise_model.basis_gates)
    h1_instruction = Instruction(embedding[1], Op.H).to_basis_gate_impl(noise_model.basis_gates)
    ry0_instruction = Instruction(embedding[0], Op.RY, params=[pi/4]).to_basis_gate_impl(noise_model.basis_gates)
    ry1_instruction =  Instruction(embedding[1], Op.RY, params=[pi/4]).to_basis_gate_impl(noise_model.basis_gates)
    
    actions = []
    if experiment_id == TwoQZeroPlusExperimentID.TWOQ:
        assert len(embedding) == 2
        
        h0_action = POMDPAction("H0", h0_instruction)
        actions.append(h0_action)
        
        h1_action = POMDPAction("H1", h1_instruction)
        actions.append(h1_action)
        
        ry0_action = POMDPAction("RY0", ry0_instruction)
        actions.append(ry0_action)
        
        
        ry1_action = POMDPAction("RY1", ry1_instruction)
        actions.append(ry1_action)
        
        if Instruction(embedding[1], Op.CNOT, control=embedding[0]) in noise_model.instructions_to_channel.keys():
            rycx01_action = POMDPAction("rycx01", ry1_instruction + [Instruction(embedding[1], Op.CNOT, control=embedding[0]), Instruction(4, Op.WRITE1)])
            actions.append(rycx01_action)
        
        if Instruction(embedding[0], Op.CNOT, control=embedding[1]) in noise_model.instructions_to_channel.keys():
            hcx10_action = POMDPAction("hcx10", h1_instruction + [Instruction(embedding[0], Op.CNOT, control=embedding[1]), Instruction(4, Op.WRITE1)])
            actions.append(hcx10_action)
        
        hcu10_action = POMDPAction("hcu10", [Instruction(4, Op.WRITE1)] + h1_instruction + 
                        # control-ry gate
                        Instruction(embedding[0], Op.CNOT, control=embedding[1]).to_basis_gate_impl(noise_model.basis_gates)+
                        Instruction(embedding[0], Op.RY, params=[-pi/4]).to_basis_gate_impl(noise_model.basis_gates) +
                        Instruction(embedding[0], Op.CNOT, control=embedding[1]).to_basis_gate_impl(noise_model.basis_gates)+
                        ry0_instruction
                        )
        # actions.append(hcu10_action)
        
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
    elif experiment_id == TwoQZeroPlusExperimentID.TWOQ2:
        hcu10_action = POMDPAction("hcu10", [Instruction(4, Op.WRITE1)] + h1_instruction + 
                        # control-ry gate
                        Instruction(embedding[0], Op.CNOT, control=embedding[1]).to_basis_gate_impl(noise_model.basis_gates)+
                        Instruction(embedding[0], Op.RY, params=[-pi/4]).to_basis_gate_impl(noise_model.basis_gates) +
                        Instruction(embedding[0], Op.CNOT, control=embedding[1]).to_basis_gate_impl(noise_model.basis_gates)+
                        ry0_instruction
                        )
        actions.append(hcu10_action)
        
    return actions

def is_dictionary_in_list(l: List[Dict[int, int]], d: Dict[int, int]):
    assert len(d) == 2
    assert 0 in d.keys()
    assert 1 in d.keys()
    for d_ in l:
        assert len(d_) == 2
        assert 0 in d_.keys()
        assert 1 in d_.keys()
        if d[0] == d_[0] and d[1] == d_[1]:
            return True
    return False

def get_selected_couplers(noise_model: NoiseModel, target: int) -> List[Tuple[int, int]]:
    answer = []
    couplers = [x[0] for x in noise_model.get_qubit_couplers(target, is_target=True)]
    if len(couplers) > 0:
        answer.append((couplers[0], target))
        answer.append((couplers[-1], target))
    
    couplers = [x[0] for x in noise_model.get_qubit_couplers(target, is_target=False)]
    if len(couplers) > 0:
        answer.append((couplers[0], target))
        answer.append((couplers[-1], target))
    
    return answer

def get_hardware_scenarios(hardware_spec: HardwareSpec, experiment_id) -> List[Dict[int, int]]:
    ''' returns hardware scenarios (embeddings) for a given hardware specification
    '''
    assert experiment_id in [TwoQZeroPlusExperimentID.TWOQ]
    noise_model = NoiseModel(hardware_spec, thermal_relaxation=False)
    answer = []
    
    selected_couplers = set()
    # we consider the most noisy couplers
    selected_couplers.add(noise_model.get_most_noisy_couplers()[0][0])
        
    # we consider the least noisy couplers
    selected_couplers.add(noise_model.get_most_noisy_couplers()[-1][0])
    
    # choose qubits according to measurement error
    pivot_qubits = get_pivot_qubits(noise_model, only_most_noisy=False)
    for pivot in pivot_qubits:
        couplers = get_selected_couplers(noise_model, pivot)
        for coupler in couplers:
            if (coupler[1], coupler[0]) not in selected_couplers:
                selected_couplers.add(coupler)
    
    for coupler in selected_couplers:
        d = {0: coupler[0], 1: coupler[1]}
        assert not is_dictionary_in_list(answer, d)
        answer.append(deepcopy(d))
        
        if Instruction(coupler[0], Op.CNOT, control=coupler[1]) not in noise_model.instructions_to_channel.keys() or Instruction(coupler[1], Op.CNOT, control=coupler[0]) not in noise_model.instructions_to_channel.keys():
            # if either of the directions does not exists, we consider the other possible embedding with this coupler
            d = {0: coupler[1], 1: coupler[0]}
            assert not is_dictionary_in_list(answer, d)
            answer.append(deepcopy(d))
        
    return answer

def twoq_guard(vertex: POMDPVertex, _: Dict[int, int], action: POMDPAction) -> bool:
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
    # arg = sys.argv[1]
    Precision.PRECISION = MAX_PRECISION
    Precision.update_threshold()
    allowed_hardware = get_allowed_hardware()
    
    experiment_id = TwoQZeroPlusExperimentID.TWOQ
    
    settings = get_project_settings()
    project_path = settings["PROJECT_PATH"]
    batches = get_num_qubits_to_hardware(WITH_THERMALIZATION, allowed_hardware)
    
    # print("Generating configuration files...")
    # generate_configs(experiment_id, min_horizon=3, max_horizon=5, allowed_hardware=allowed_hardware)
    
    # print("generating embedding files...")
    # for num_qubits in batches.keys():
    #     config_path = get_config_path(experiment_id, num_qubits)
    #     generate_embeddings(experiment_id, num_qubits, get_hardware_embeddings=get_hardware_scenarios)
        
    for num_qubits in batches.keys():
    # # num_qubits = arg
        config_path = get_config_path(experiment_id, num_qubits)
        generate_pomdps(experiment_id, num_qubits, get_experiments_actions, ZeroPlusInstance, guard=twoq_guard, set_hidden_index=True)

    
    
    
        