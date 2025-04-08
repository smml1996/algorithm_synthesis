from copy import deepcopy
from enum import Enum
from math import pi
import numpy as np
import os, sys
from typing import Dict, List, Tuple
sys.path.append(os.getcwd()+"/..")

from utils import Precision, find_enum_object
from ibm_noise_models import HardwareSpec, Instruction, NoiseModel
from qstates import QuantumState
from qpu_utils import BasisGates, Op
from cmemory import ClassicalState, cread
from pomdp import POMDPAction, POMDPVertex
import qmemory
from experiments_utils import TwoQZeroPlusExperimentID, check_files, generate_algs_vs_file, generate_configs, generate_diff_algorithms_file, generate_embeddings, generate_mc_guarantees_file, generate_pomdps, get_allowed_hardware, get_config_path, get_num_qubits_to_hardware, get_project_settings
from bitflip import get_pivot_qubits
    
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
        if experiment_id in [TwoQZeroPlusExperimentID.TWOQ, TwoQZeroPlusExperimentID.TWOQ2, TwoQZeroPlusExperimentID.HCXH,TwoQZeroPlusExperimentID.HCXH2, TwoQZeroPlusExperimentID.ENTSWAP]:
            self.qubits_used = [self.embedding[0], self.embedding[1]]
            # check embedding
            assert len(self.embedding.keys()) == 2 # 2 + 2 qubit for hidden indices
        elif experiment_id in [TwoQZeroPlusExperimentID.ONEQT]:
            self.qubits_used = [self.embedding[0]]
            assert len(self.embedding.keys()) == 1
        else:
            raise Exception("missing setup for experiment", experiment_id)
        self.get_initial_distribution()
        
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
    
    def get_reward(self, vertex: POMDPVertex) -> float:     
        return int(cread(vertex.classical_state, 0) == vertex.hidden_index)
    
def get_experiments_actions(noise_model: NoiseModel, embedding, experiment_id):
    assert isinstance(noise_model, NoiseModel)
    assert isinstance(experiment_id, TwoQZeroPlusExperimentID)
    
    # some instructions
    h0_instruction = Instruction(embedding[0], Op.H).to_basis_gate_impl(noise_model.basis_gates)
    ry0_instruction = Instruction(embedding[0], Op.RY, params=[pi/4]).to_basis_gate_impl(noise_model.basis_gates)
    
    if len(embedding.keys()) > 1 :
        h1_instruction = Instruction(embedding[1], Op.H).to_basis_gate_impl(noise_model.basis_gates)
        ry1_instruction =  Instruction(embedding[1], Op.RY, params=[pi/4]).to_basis_gate_impl(noise_model.basis_gates)
    
    actions = []
    if experiment_id == TwoQZeroPlusExperimentID.ONEQT:
        ry0_action = POMDPAction("RY0", ry0_instruction)
        actions.append(ry0_action)
        
        for i in [2, 5, 10]:
            ry_instruction = Instruction(embedding[0], Op.RY, params=[np.radians(i)]).to_basis_gate_impl(noise_model.basis_gates)
            actions.append(POMDPAction(f"RY{i}", ry_instruction))
            
            ry_instruction = Instruction(embedding[0], Op.RY, params=[np.radians(i)]).to_basis_gate_impl(noise_model.basis_gates)
            actions.append(POMDPAction(f"RY-{i}", ry_instruction))
        
        meas_action = POMDPAction("MEAS", [
            Instruction(embedding[0], Op.MEAS, real_target=0),
            Instruction(1, Op.WRITE1)
        ])
        actions.append(meas_action)
        
        DETERMINE0 = POMDPAction("IS0", [Instruction(0, Op.WRITE0), Instruction(2, Op.WRITE1)])
        DETERMINEPlus = POMDPAction("ISPlus", [Instruction(0, Op.WRITE1), Instruction(2, Op.WRITE1)])
        actions.append(DETERMINE0)
        actions.append(DETERMINEPlus)
    elif experiment_id in [TwoQZeroPlusExperimentID.HCXH, TwoQZeroPlusExperimentID.HCXH2]:
        actions.append(POMDPAction("RY0", ry0_instruction + [
            Instruction(2, Op.WRITE1)
        ]))
        
        h1_action = POMDPAction("H1", h1_instruction)
        actions.append(h1_action)
        x1_action = POMDPAction("X1", Instruction(embedding[1], Op.X).to_basis_gate_impl(noise_model.basis_gates))
        actions.append(x1_action)
        
        if Instruction(embedding[1], Op.CNOT, control=embedding[0]) in noise_model.instructions_to_channel.keys():
            cx01_action = POMDPAction("CX01", [Instruction(3, Op.WRITE1), 
                                               Instruction(embedding[1], Op.CNOT, control=embedding[0])])
            actions.append(cx01_action)
        
        if Instruction(embedding[0], Op.CNOT, control=embedding[1]) in noise_model.instructions_to_channel.keys():
            cx10_action = POMDPAction("CX10", [Instruction(3, Op.WRITE1), Instruction(embedding[0], Op.CNOT, control=embedding[1])])
            actions.append(cx10_action)
        
        if experiment_id == TwoQZeroPlusExperimentID.HCXH:
            meas0_action = POMDPAction("MEAS0", [Instruction(embedding[0], Op.MEAS, real_target=0),
                                                Instruction(embedding[1], Op.MEAS, real_target=1),
                                                Instruction(4, Op.WRITE1)])
            actions.append(meas0_action)
        else:
            assert experiment_id == TwoQZeroPlusExperimentID.HCXH2
            meas0_action = POMDPAction("MEAS0", [Instruction(embedding[0], Op.MEAS, real_target=0),
                                                Instruction(4, Op.WRITE1)])
            meas1_action = POMDPAction("MEAS1", [Instruction(embedding[1], Op.MEAS, real_target=1)])
            actions.append(meas0_action)
            actions.append(meas1_action)
        
        DETERMINE0 = POMDPAction("IS0", [Instruction(0, Op.WRITE0), Instruction(5, Op.WRITE1)])
        DETERMINEPlus = POMDPAction("ISPlus", [Instruction(0, Op.WRITE1), Instruction(5, Op.WRITE1)])
        
        actions.append(DETERMINE0)
        actions.append(DETERMINEPlus)
    elif experiment_id == TwoQZeroPlusExperimentID.TWOQ:
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
        actions.append(POMDPAction("RY0", ry0_instruction + [
            Instruction(4, Op.WRITE1)
        ]))
        
        count_cnot_directions = 0
        if Instruction(embedding[1], Op.CNOT, control=embedding[0]) in noise_model.instructions_to_channel.keys():
            count_cnot_directions = 1
            rycx01_action = POMDPAction("rycx01", [Instruction(5, Op.WRITE1), Instruction(7, Op.WRITE1)] + ry1_instruction + [Instruction(embedding[1], Op.CNOT, control=embedding[0])] +  ry1_instruction)
            actions.append(rycx01_action)
        
        if Instruction(embedding[0], Op.CNOT, control=embedding[1]) in noise_model.instructions_to_channel.keys():
            count_cnot_directions += 1
            hcx10_action = POMDPAction("hcx10", [Instruction(5, Op.WRITE1), Instruction(7, Op.WRITE1)]+ h1_instruction + [Instruction(embedding[0], Op.CNOT, control=embedding[1])] + h1_instruction)
            actions.append(hcx10_action)
        
        meas0_action = POMDPAction("MEAS0", [Instruction(embedding[0], Op.MEAS, real_target=0),Instruction(2, Op.WRITE1), Instruction(6, Op.WRITE1), Instruction(5, Op.WRITE0)])
        meas1_action = POMDPAction("MEAS1", [Instruction(embedding[1], Op.MEAS, real_target=1),Instruction(2, Op.WRITE1), Instruction(5, Op.WRITE0)])
        actions.append(meas0_action)
        actions.append(meas1_action)
        
        DETERMINE0 = POMDPAction("IS0", [Instruction(0, Op.WRITE0), Instruction(3, Op.WRITE1)])
        DETERMINEPlus = POMDPAction("ISPlus", [Instruction(0, Op.WRITE1), Instruction(3, Op.WRITE1)])
        actions.append(DETERMINE0)
        actions.append(DETERMINEPlus)
    elif experiment_id == TwoQZeroPlusExperimentID.ENTSWAP:
        assert Instruction(embedding[1], Op.CNOT, control=embedding[0]) in noise_model.instructions_to_channel.keys()
        
        #     0       1         2         3      4           5
        # | meas0 | meas1 | guess_made | RY0? | RY1? | multiqubit_gate
        
        actions.append(POMDPAction("RY0", ry0_instruction + [
            Instruction(3, Op.WRITE1)
        ]))
        
        actions.append(POMDPAction("RY1", ry1_instruction + [
            Instruction(4, Op.WRITE1)
        ]))
    
        cx01_action = POMDPAction("CX01", [
            Instruction(embedding[1], Op.CNOT, control=embedding[0]),
            Instruction(5, Op.WRITE1)
            ])
        actions.append(cx01_action)
        
        if Instruction(embedding[0], Op.CNOT, control=embedding[1]) in noise_model.instructions_to_channel.keys():
            swap1_action = POMDPAction("SWAP10", [
                Instruction(embedding[0], Op.CNOT, control=embedding[1]),
                Instruction(embedding[1], Op.CNOT, control=embedding[0]),
                Instruction(embedding[0], Op.CNOT, control=embedding[1]),
                Instruction(5, Op.WRITE1)
                ])
            actions.append(swap1_action)
            
            swap2_action = POMDPAction("SWAP01", [
                Instruction(embedding[1], Op.CNOT, control=embedding[0]),
                Instruction(embedding[0], Op.CNOT, control=embedding[1]),
                Instruction(embedding[1], Op.CNOT, control=embedding[0]),
                Instruction(5, Op.WRITE1)])
            actions.append(swap2_action)
        
        meas0_action = POMDPAction("MEAS0", [Instruction(embedding[0], Op.MEAS, real_target=0)])
        meas1_action = POMDPAction("MEAS1", [Instruction(embedding[1], Op.MEAS, real_target=1)])
        actions.append(meas0_action)
        actions.append(meas1_action)
        
        DETERMINE0 = POMDPAction("IS0", [Instruction(0, Op.WRITE0), # c[0] == 0 we think is state |0>
                                         Instruction(2, Op.WRITE1) # if c[2] == 1 means we have already tried to make a guess
                                         ])
        DETERMINEPlus = POMDPAction("ISPlus", [Instruction(0, Op.WRITE1), # c[0] == 1 we think is state |+>
                                               Instruction(2, Op.WRITE1)] # if c[2] == 1 means we have already tried to make a guess
                                    )
        actions.append(DETERMINE0)
        actions.append(DETERMINEPlus)
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
    noise_model = NoiseModel(hardware_spec, thermal_relaxation=False)
    answer = []
    
    if experiment_id in [TwoQZeroPlusExperimentID.ONEQT]:
        qubits = set()
        # choose qubits according to measurement error
        pivot_qubits = get_pivot_qubits(noise_model, only_most_noisy=False, with_indegree=False)
        for q in pivot_qubits:
            qubits.add(q)
        
        instruction_ry = Instruction(0, Op.RY, params=[pi/4]).to_basis_gate_impl(noise_model.basis_gates)
        
        ops = [instruction.op for instruction in instruction_ry]
        for op in ops:
            most_noisy = noise_model.get_most_noisy_qubit(op, top=1)
            if most_noisy[0][0] < 0.99:
                qubits.add(most_noisy[0][1])

        for qubit in qubits:
            answer.append({0: qubit})
        
    elif experiment_id in [TwoQZeroPlusExperimentID.TWOQ, TwoQZeroPlusExperimentID.TWOQ2, TwoQZeroPlusExperimentID.HCXH,TwoQZeroPlusExperimentID.HCXH2]:
        # choose qubits according to measurement error
        pivot_qubits = get_pivot_qubits(noise_model, only_most_noisy=False)
        selected_couplers = set()
        # we consider the most noisy couplers
        selected_couplers.add(noise_model.get_most_noisy_couplers()[0][0])
            
        # we consider the least noisy couplers
        selected_couplers.add(noise_model.get_most_noisy_couplers()[-1][0])
        
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
    else:
        raise Exception("no hardware scenarios specified for experiment", experiment_id)
    return answer

def hcxh_guard(vertex: POMDPVertex, _: Dict[int, int], action: POMDPAction) -> bool:
    classical_state = vertex.classical_state
    if cread(classical_state, 5) == 1:
        return False
    
    if cread(classical_state, 4) == 1:
        return action.name in ["IS0", "ISPlus"]
    
    if cread(classical_state, 2) == 1:
        return action.name not in ["RY0"]
    
    if cread(classical_state, 3) == 1:
        return action.name not in ["CX10", "CX01"]    
    return True
        
def twoq_guard(vertex: POMDPVertex, _: Dict[int, int], action: POMDPAction) -> bool:
    classical_state = vertex.classical_state
    if cread(classical_state, 3) == 1:
        # we have already executed either Determine0 or DeterminePlus
        return False
    
    if cread(classical_state, 2) == 1:
        return action.name in ["IS0", "ISPlus"]
    
    if cread(classical_state, 4) == 1:
        return not (action.name in ["hcu10", "hcx10", "rycx01"])
    
    return True

def twoq2_guard(vertex: POMDPVertex, _: Dict[int, int], action: POMDPAction) -> bool:
    classical_state = vertex.classical_state
    
    if cread(classical_state, 3) == 1:
        # we have already determine if state is |0> or |+>
        return False
    
    if action.name == "RY0":
        if cread(classical_state, 4) == 1:
            # we allow only 1 rotation gate
            return False
        
        if cread(classical_state, 6) == 1:
            # measurement in the first qubit has already happened, there is no more information at qubit 0 (rotation gate is useless now)
            return False
    elif action.name in ["rycx01", "hcx10"]:
        if cread(classical_state, 5) == 1:
            return False
        
        if cread(classical_state, 6) == 1:
            # measurement to first qubit already performed (no info there)
            return False
    elif action.name == "MEAS0":
        return True
    elif action.name == "MEAS1":
        if cread(classical_state, 7) == 0:
            # at least 1 multiqubit gate has been performed
            return False
    elif action.name in ["IS0", "ISPlus"]:
        if cread(classical_state, 2) == 0:
            return False
    else:
        raise Exception("Invalid action", action.name)
    return True
    
    

def oneqt_guard(vertex: POMDPVertex, _: Dict[int, int], action: POMDPAction) -> bool:
    classical_state = vertex.classical_state
    
    if cread(classical_state, 2) == 1:
        # we have already executed either Determine0 or DeterminePlus
        return False
    
    if cread(classical_state, 1) == 1:
        return action.name in ["MEAS", "IS0", "ISPlus"]
    
    return True
    
    

def set_precision(experiment_id):
    if experiment_id in [TwoQZeroPlusExperimentID.TWOQ,TwoQZeroPlusExperimentID.HCXH2]:
        Precision.PRECISION = 5
    elif experiment_id in [TwoQZeroPlusExperimentID.ONEQT, TwoQZeroPlusExperimentID.TWOQ2, TwoQZeroPlusExperimentID.HCXH]:
        Precision.PRECISION = 8
    else:
        raise Exception("Could not set precision for", experiment_id)
    Precision.update_threshold()

def get_min_max_horizon(experiment_id) -> Tuple[int, int]:
    if experiment_id == TwoQZeroPlusExperimentID.ONEQT:
        return 2, 3
    elif experiment_id == TwoQZeroPlusExperimentID.TWOQ:
        return 3, 5
    elif experiment_id == TwoQZeroPlusExperimentID.TWOQ2:
        return 2, 5
    elif experiment_id in [TwoQZeroPlusExperimentID.HCXH,TwoQZeroPlusExperimentID.HCXH2]:
        return 2, 6
    else:
        raise Exception("could not retrieve min. and max. horizon for experiment", experiment_id)
    
def get_guard(experiment_id):
    if experiment_id == TwoQZeroPlusExperimentID.ONEQT:
        return oneqt_guard
    elif experiment_id == TwoQZeroPlusExperimentID.TWOQ:
        return twoq_guard
    elif experiment_id == TwoQZeroPlusExperimentID.TWOQ2:
        return twoq2_guard
    elif experiment_id in [TwoQZeroPlusExperimentID.HCXH,TwoQZeroPlusExperimentID.HCXH2]:
        return hcxh_guard
    else:
        raise Exception("could not retireve guard for experiment", experiment_id)

def get_thermalization_setup(experiment_id) -> bool:
    if experiment_id in [TwoQZeroPlusExperimentID.ONEQT]:
        return True
    elif experiment_id in [TwoQZeroPlusExperimentID.TWOQ, TwoQZeroPlusExperimentID.TWOQ2, TwoQZeroPlusExperimentID.HCXH,TwoQZeroPlusExperimentID.HCXH2]:
        return False
    else:
        raise Exception("Could not get thermalization setup for experiment", experiment_id)

if __name__ == "__main__":
    settings = get_project_settings()
    project_path = settings["PROJECT_PATH"]
    
    experiment_name = sys.argv[1]
    process_name = sys.argv[2]
    batch_name = None
    if len(sys.argv) > 3:
        batch_name = sys.argv[3]
    
    experiment_id = find_enum_object(experiment_name, TwoQZeroPlusExperimentID)
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
                generate_pomdps(experiment_id, num_qubits, get_experiments_actions, ZeroPlusInstance, guard=get_guard(experiment_id), set_hidden_index=True, WITH_THERMALIZATION=with_thermalization)
        else:
            config_path = get_config_path(experiment_id, batch_name)
            generate_pomdps(experiment_id, batch_name, get_experiments_actions, ZeroPlusInstance, guard=get_guard(experiment_id), set_hidden_index=True, WITH_THERMALIZATION=with_thermalization)
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
    

    
    
    
        