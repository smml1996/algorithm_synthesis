from copy import deepcopy
from enum import Enum
from math import pi
import os, sys
from typing import Dict, List
sys.path.append(os.getcwd()+"/..")

from utils import are_matrices_equal, Precision
from ibm_noise_models import HardwareSpec, Instruction, MeasChannel, NoiseModel
from qstates import QuantumState
from qpu_utils import Op
from cmemory import ClassicalState, cread
from pomdp import POMDPAction, POMDPVertex
import qmemory
from experiments_utils import ReadoutNoise, rho_qubit0, rho_qubit1, run_bellmaneq
from experiments_utils import check_files, generate_algs_vs_file, generate_configs, generate_diff_algorithms_file, generate_embeddings, generate_mc_guarantees_file, generate_pomdps, get_config_path, get_embeddings_path, get_num_qubits_to_hardware, get_project_settings, bell_state_pts, load_embeddings

MAX_PRECISION = 10
WITH_THERMALIZATION = False

class ZeroPlusExperimentID(Enum):
    ONEQ = "oneq"
    ONEQ10PI = "oneq10pi"
    TWOQ1 = "twoq1" # cx(1,2), H(2), T(2), TD(2), meas(2).
    TWOQCH = "twoqch" # ch(1,2), meas(1), meas(2), x(2)
    @property
    def exp_name(self):
        return "zeroplus"

class ZeroPlusInstance:
    def __init__(self, embedding, experiment_id: ZeroPlusExperimentID):
        """_summary_

        Args:
            num_qubits (int): _description_
            instruction_set (List[Instruction]): _description_
            embedding (Dict[int, int]): a mapping from logical qubits to physical qubits
        """
        assert isinstance(experiment_id, ZeroPlusExperimentID)
        self.embedding = embedding
        self.experiment_id = experiment_id
        self.initial_distribution = None
        # check embedding
        assert 0 in self.embedding.keys()
        assert 1 in self.embedding.keys()
        self.qubits_used = [self.embedding[0], self.embedding[1]]
        if self.experiment_id in [ZeroPlusExperimentID.ONEQ, ZeroPlusExperimentID.ONEQ10PI]:
            assert len(self.embedding.keys()) == 3
            self.hidden_index = 1
            self.remove_qubits = [self.embedding[0]]
        elif self.experiment_id in [ZeroPlusExperimentID.TWOQ1, ZeroPlusExperimentID.TWOQCH]:
            assert 2 in self.embedding.keys()
            assert len(self.embedding.keys()) == 4 # 2 + 2 qubit for hidden indices
            self.hidden_index = 2
            self.qubits_used.append(self.embedding[2])
            self.remove_qubits = [self.embedding[0],self.embedding[1]]
        else:
            raise Exception(f"Setup correctly experiment {self.experiment_id}")
        self.get_initial_distribution()
        

    def get_initial_distribution(self):
        self.initial_distribution = []
        initial_cs = ClassicalState()
        
        H0 = Instruction(self.embedding[0], Op.H).get_gate_data()
        XHidden = Instruction(self.embedding[self.hidden_index], Op.X).get_gate_data()
        
        zero = QuantumState(0, qubits_used=self.qubits_used)
        self.initial_distribution.append(((zero, initial_cs), 0.5))
        
        plus = qmemory.handle_write(zero, H0)
        plus = qmemory.handle_write(plus, XHidden)
        self.initial_distribution.append(((plus, initial_cs), 0.5))
        
    def get_reward(self, hybrid_state) -> float:
        qs, cs = hybrid_state        
        assert isinstance(qs, QuantumState)
        current_rho = qs.multi_partial_trace(remove_indices=self.remove_qubits)
        
        # we should get a 1 qubit density matrix
        assert len(current_rho) == 2
        assert len(current_rho[0]) == len(current_rho[1]) 
        assert len(current_rho[0]) == 2
        assert not (cread(cs, self.embedding[self.hidden_index]) == 1 and cread(cs, self.embedding[self.hidden_index+1]) == 1) 
        if are_matrices_equal(current_rho, rho_qubit0):
            return int(cread(cs, self.embedding[self.hidden_index]) == 1 and cread(cs, self.embedding[self.hidden_index+1]) == 0)
        assert are_matrices_equal(current_rho, rho_qubit1)
        return int(cread(cs, self.embedding[self.hidden_index]) == 0 and cread(cs, self.embedding[self.hidden_index + 1]) == 1)
    
def get_experiments_actions(noise_model, embedding, experiment_id):
    assert isinstance(noise_model, NoiseModel)
    assert isinstance(experiment_id, ZeroPlusExperimentID)
    
    num_qubits = len(embedding) - 2
    hidden_index = num_qubits
    actions = []

    if experiment_id == ZeroPlusExperimentID.ONEQ:
        for i in range(num_qubits):
            # single qubit instructions
            ops = [Op.H, Op.MEAS, Op.X, Op.Z]
            for op in ops:
                actions.append(POMDPAction(f"{op.name}{i}", Instruction(embedding[i], op).to_basis_gate_impl(noise_model.basis_gates)))
            
            # all possible CX instructions
            for j in range(0, num_qubits):
                if i != j:
                    cx_instruction = Instruction(embedding[j], Op.CNOT, embedding[i])
                    if cx_instruction in noise_model.instructions_to_channel.keys():
                        actions.append(POMDPAction(f"CX{i}{j}", [cx_instruction]))
    elif experiment_id == ZeroPlusExperimentID.ONEQ10PI:
        meas_instruction = Instruction(embedding[0], Op.MEAS).to_basis_gate_impl(noise_model.basis_gates)
        actions.append(POMDPAction("MEAS", meas_instruction))
        
        for i in range(4, 6):
            rz_instruction = Instruction(embedding[0], Op.RZ, params=[pi/i]).to_basis_gate_impl(noise_model.basis_gates)
            # nrz_instruction = Instruction(embedding[0], Op.RZ, params=[-pi/i]).to_basis_gate_impl(noise_model.basis_gates)
            actions.append(POMDPAction(f"RZ{i}", rz_instruction))
            # actions.append(POMDPAction(f"RZ-{i}", nrz_instruction))
            
            ry_instruction = Instruction(embedding[0], Op.RY, params=[pi/i]).to_basis_gate_impl(noise_model.basis_gates)
            # nry_instruction = Instruction(embedding[0], Op.RY, params=[-pi/i]).to_basis_gate_impl(noise_model.basis_gates)
            actions.append(POMDPAction(f"RY{i}", ry_instruction))
            # actions.append(POMDPAction(f"RY-{i}", nry_instruction))
        print("Total actions:", len(actions))
            
    elif experiment_id == ZeroPlusExperimentID.TWOQ1:
        cx_instruction = Instruction(embedding[1], Op.CNOT, embedding[0])
        actions.append(POMDPAction("CX01", [cx_instruction]))
        
        h_instruction = Instruction(embedding[1], Op.H).to_basis_gate_impl(noise_model.basis_gates)
        actions.append(POMDPAction("H1", h_instruction))
        
        t_instruction = Instruction(embedding[1], Op.T).to_basis_gate_impl(noise_model.basis_gates)
        actions.append(POMDPAction("T1", t_instruction))
        
        td_instruction = Instruction(embedding[1], Op.TD).to_basis_gate_impl(noise_model.basis_gates)
        actions.append(POMDPAction("TD1", td_instruction))
        
        meas_instruction = Instruction(embedding[1], Op.MEAS).to_basis_gate_impl(noise_model.basis_gates)
        actions.append(POMDPAction("MEAS1", meas_instruction))
    elif experiment_id == ZeroPlusExperimentID.TWOQCH:
        ch_instruction = Instruction(embedding[1], Op.CH, embedding[0]).to_basis_gate_impl(noise_model.basis_gates)
        actions.append(POMDPAction("CH01", ch_instruction))
        
        x_instruction = Instruction(embedding[1], Op.X).to_basis_gate_impl(noise_model.basis_gates)
        actions.append(POMDPAction("X1", x_instruction))
        
        meas_instruction = Instruction(embedding[1], Op.MEAS).to_basis_gate_impl(noise_model.basis_gates)
        actions.append(POMDPAction("MEAS1", meas_instruction))
        meas_instruction = Instruction(embedding[0], Op.MEAS).to_basis_gate_impl(noise_model.basis_gates)
        actions.append(POMDPAction("MEAS0", meas_instruction))
    DETERMINE0 = POMDPAction("IS0", [Instruction(embedding[hidden_index], Op.WRITE1)])
    DETERMINEPlus = POMDPAction("ISPlus", [Instruction(embedding[hidden_index+1], Op.WRITE1)])
    actions.append(DETERMINE0)
    actions.append(DETERMINEPlus)
        
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
    if experiment_id in [ZeroPlusExperimentID.ONEQ, ZeroPlusExperimentID.ONEQ10PI]:
       for i in pivot_qubits:
           embedding = dict()
           embedding[0] = i
           embedding[1] = get_unused_qubit(noise_model, used_qubits=[i])
           embedding[2] = get_unused_qubit(noise_model, used_qubits=[i, embedding[1]])
           answer.append(deepcopy(embedding))
    elif experiment_id in [ZeroPlusExperimentID.TWOQ1, ZeroPlusExperimentID.TWOQCH]:
        most_noisy_coupler = noise_model.get_most_noisy_couplers()[0][0]
        embedding = dict()
        control = most_noisy_coupler[0]
        target = most_noisy_coupler[1]
        embedding[0] = control
        embedding[1] = target
        embedding[2] = get_unused_qubit(noise_model, used_qubits=[control, target])
        embedding[3] = get_unused_qubit(noise_model, used_qubits=[control, target, embedding[2]])
        answer.append(deepcopy(embedding))
    else:
        raise Exception(f"get_hardware_scenarios for experiment {experiment_id} not implemented")
    return answer
    
def halt_guard(vertex: POMDPVertex, embedding: Dict[int, int], action: POMDPAction) -> bool:
    cs = vertex.classical_state
    hidden_index = len(embedding.keys()) - 2
    return cread(cs, embedding[hidden_index]) == 0 and cread(cs, embedding[hidden_index+1]) == 0
    
if __name__ == "__main__":
    arg = sys.argv[1]
    
    if arg == "oneq":
        experiment_id = ZeroPlusExperimentID.ONEQ
    elif arg == "twoq1":
        experiment_id = ZeroPlusExperimentID.TWOQ1
    elif arg == "twoqch":
        experiment_id = ZeroPlusExperimentID.TWOQCH
    elif arg == "oneq10":
        experiment_id = ZeroPlusExperimentID.ONEQ10PI
    else:
        raise Exception("argument does not match anything")
    
    Precision.PRECISION = MAX_PRECISION
    Precision.update_threshold()
    settings = get_project_settings()
    project_path = settings["PROJECT_PATH"]
    
    print(experiment_id)
    print("generating config files")
    generate_configs(experiment_id=experiment_id, min_horizon=3, max_horizon=4)
    
    print("generating embedding files...")
    batches = get_num_qubits_to_hardware(WITH_THERMALIZATION)
    for num_qubits in batches.keys():
        config_path = get_config_path(experiment_id, num_qubits)
        generate_embeddings(experiment_id, num_qubits, get_hardware_embeddings=get_hardware_scenarios)
    
    project_settings = get_project_settings()
    for num_qubits in batches.keys():
        config_path = get_config_path(experiment_id, num_qubits)
        generate_pomdps(experiment_id, num_qubits, get_experiments_actions, ZeroPlusInstance, guard=halt_guard)
        # run_bellmaneq(project_settings, config_path)
 
        