import os, sys
sys.path.append(os.getcwd()+"/..")

from experiments_utils import directory_exists, generate_configs, generate_embeddings, get_config_path, get_embeddings_path, get_num_qubits_to_hardware, get_project_path, get_project_settings


from cmath import isclose
from copy import deepcopy
import time
from typing import Any, Dict, List
import json

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from algorithm import AlgorithmNode, execute_algorithm
from cmemory import ClassicalState, cread, cwrite
from pomdp import POMDP, POMDPAction, POMDPVertex, build_pomdp
import qmemory
from qpu_utils import GateData, Op, BasisGates
from utils import Queue, are_matrices_equal, find_enum_object, get_index, is_matrix_in_list, Precision
sys.path.append(os.getcwd()+"/..")

from qstates import QuantumState
from ibm_noise_models import Instruction, MeasChannel, NoiseModel, get_ibm_noise_model, HardwareSpec, ibm_simulate_circuit, load_config_file
import numpy as np
from math import pi   
from enum import Enum

from bitflip import MAX_PRECISION, get_hardware_embeddings as get_hardware_embeddings_z
from phaseflip import get_hardware_embeddings as get_hardware_embeddings_x
from ghz import get_hardware_embeddings as get_hardware_embeddings_ghz
from bitflip import BitflipExperimentID
from phaseflip import PhaseflipExperimentID


BFS_DISTANCE = 2
WITH_THERMALIZATION = False
LIMIT_EMBEDDINGS = 2

class RouteExperimentID(Enum):
    EXP1 = "exp1"
    
class RouteInstance:
    def __init__(self, embedding):
        self.embedding = embedding
        self.initial_state = None
        self.get_initial_states()
        self.target_index = embedding[1]
        self.processed_states = []
        
    def get_initial_states(self):
        self.initial_state = []
        initial_cs = ClassicalState()
        initial_cs = cwrite(initial_cs, Op.WRITE1, self.embedding[0])
        
        RX = Instruction(self.embedding[0], Op.RX, params=[pi/3]).get_gate_data()
        state = QuantumState(0, qubits_used=list(self.embedding.values()))
        state = qmemory.handle_write(state, RX) # state susceptible to x and z errors
        self.initial_state.append((state, initial_cs))
    
    def get_remove_indices(self, except_index):
        answer = []
        for (key, val) in self.embedding.items():
            if key != except_index:
                answer.append(val)
        return answer
    
    def get_reward(self, hybrid_state) -> float:
        qs , cs = hybrid_state
        if not cread(cs, self.embedding[1]):
            return 0.00
        for (state, val) in self.processed_states:
            if state == qs:
                return val
        remove_indices =  self.get_remove_indices(0)
        target_rho = qs.multi_partial_trace(remove_indices=remove_indices)
        remove_indices = self.get_remove_indices(1)
        current_rho = qs.multi_partial_trace(remove_indices=remove_indices)
        assert len(current_rho) == len(target_rho)
        assert len(current_rho) == 2
        assert len(current_rho[0]) ==  len(target_rho[0])
        assert len(current_rho[0]) == 2
        if are_matrices_equal(current_rho, target_rho):
            self.processed_states.append((qs, 1.00))
            return 1.00
        else:
            self.processed_states.append((qs, 0.00))
            return 0.00
        
def get_bfs_distance(noise_model, q1, q2):
    visited = set()
    q = Queue()
    q.push((q1, 0))
    visited.add(q1)
    
    g = noise_model.digraph
    
    while not q.is_empty():
        curr_q, curr_depth = q.pop()
        for succ in g[curr_q]:
            if succ == q2:
                return curr_depth + 1
            if curr_depth < BFS_DISTANCE:
                if succ not in visited:
                    q.push((succ, curr_depth +1))
            visited.add(succ)
                
    return BFS_DISTANCE + 1
        
def is_embedding_equal(e1, e2):
    for i in range(2):
        if e1[i] != e2[i]:
            return False
    return True

def is_repeated_embedding(all_embeddings, current_embedding):
    for embedding in all_embeddings:
        if is_embedding_equal(embedding, current_embedding):
            return True
    return False

def is_embedding_feasible(noise_model, current_embedding):
    if len(current_embedding.keys()) >8:
        return False
    
    visited = set()
    q = Queue()
    q.push((current_embedding[0], 0))
    visited.add(current_embedding[0])
    
    g = noise_model.digraph
    
    while not q.is_empty():
        curr_q, curr_depth = q.pop()
        for succ in g[curr_q]:
            if curr_depth < BFS_DISTANCE:
                if succ not in visited:
                    if len(g[succ]) > 4:
                        return False
                    q.push((succ, curr_depth +1))
            visited.add(succ)
    
    return True

def get_embedding(noise_model, q1, q2):
    current_index = 1
    answer = dict()
    answer[0] = q1
    answer[1] = q2
    
    visited = set()
    q = Queue()
    q.push((q1, 0))
    visited.add(q1)
    current_index = 2
    
    g = noise_model.digraph
    
    while not q.is_empty():
        curr_q, curr_depth = q.pop()
        for succ in g[curr_q]:
            if succ != q2:
                if succ not in answer.values():
                    answer[current_index] = succ
                    current_index += 1
            if curr_depth < BFS_DISTANCE:
                if succ not in visited:
                    q.push((succ, curr_depth +1))
            visited.add(succ)
                
    return answer

def get_hardware_embeddings(backend: HardwareSpec, **kwargs):
    answer = []
    noise_model = NoiseModel(backend, thermal_relaxation=WITH_THERMALIZATION)
    assert noise_model.num_qubits >= 0
    
    ghz_embeddings = get_hardware_embeddings_ghz(backend)
    embeddings_bitflip = get_hardware_embeddings_z(backend, experiment_id=BitflipExperimentID.IPMA2)
    embeddings_phaseflip = get_hardware_embeddings_x(backend, experiment_id=PhaseflipExperimentID.IPMA)
    
    for ghz_embedding in ghz_embeddings:
        assert len(ghz_embedding.keys()) == 3
        for q1 in ghz_embedding.values():
            # bitflip feasability
            for bf_embedding in embeddings_bitflip:
            
                if get_bfs_distance(noise_model, q1, bf_embedding[0]) <= BFS_DISTANCE:
                    current_embedding = get_embedding(noise_model, q1, bf_embedding[0])
                    if not is_repeated_embedding(answer, current_embedding):
                        if is_embedding_feasible(noise_model, current_embedding):
                            answer.append(current_embedding)
                if get_bfs_distance(noise_model, q1, bf_embedding[1])<= BFS_DISTANCE:
                    current_embedding = get_embedding(noise_model, q1, bf_embedding[1])
                    if not is_repeated_embedding(answer, current_embedding):
                        if is_embedding_feasible(noise_model, current_embedding):
                            answer.append(current_embedding)
            
            # phaseflip feasability     
            for phase_embedding in embeddings_phaseflip:
                if get_bfs_distance(noise_model, q1, phase_embedding[0])<= BFS_DISTANCE:
                    current_embedding = get_embedding(noise_model, q1, phase_embedding[0])
                    if not is_repeated_embedding(answer, current_embedding):
                        if is_embedding_feasible(noise_model, current_embedding):
                            answer.append(current_embedding)
                if get_bfs_distance(noise_model, q1, phase_embedding[1])<= BFS_DISTANCE:
                    current_embedding = get_embedding(noise_model, q1, phase_embedding[1])
                    if not is_repeated_embedding(answer, current_embedding):
                        if is_embedding_feasible(noise_model, current_embedding):
                            answer.append(current_embedding)
                        
    return answer[:LIMIT_EMBEDDINGS]

def guard(vertex: POMDPVertex, embedding: Dict[int, int], action: POMDPAction):
    q_instruction = action.instruction_sequence[0]
    assert q_instruction.op == Op.CNOT
    control = q_instruction.control
    c_memory = vertex.classical_state
    return cread(c_memory, control)


def get_experiments_actions(noise_model: NoiseModel, embedding: Dict[int,int], experiment_id: RouteExperimentID):
    initial_qubit = embedding[0]
    answer = [] # list of POMDPAction(s)
    # reverse embedding
    rev_embedding = dict()
    for (key, val) in embedding.items():
        rev_embedding[val] = key
        
    assert experiment_id == RouteExperimentID.EXP1
    
    digraph = noise_model.digraph
    visited = set()
    q = Queue()
    q.push((initial_qubit, 0))
    visited.add(initial_qubit)
    
    while not q.is_empty():
        control, current_depth = q.pop()
        assert control in rev_embedding.keys()
        v_control = rev_embedding[control]
        
        if current_depth < BFS_DISTANCE:
            for target in digraph[control]:
                assert target in rev_embedding.keys()
                q_instruction = Instruction(target, Op.CNOT, control=control)
                v_target= rev_embedding[target]
                c0_instruction = Instruction(control, Op.WRITE0)
                c1_instruction = Instruction(target, Op.WRITE1)
                instruction_sequence = [q_instruction, c0_instruction, c1_instruction]
                answer.append(POMDPAction(f"CX{v_control}_{v_target}", instruction_sequence))
                if target not in visited:
                    q.push((target, current_depth + 1))
                visited.add(target)
    return answer
                
            
def generate_pomdp(experiment_id: PhaseflipExperimentID, hardware_spec: HardwareSpec, 
                embedding: Dict[int, int], pomdp_write_path: str, return_pomdp=False):
    noise_model = NoiseModel(hardware_spec, thermal_relaxation=WITH_THERMALIZATION)
    swap_instance = RouteInstance(embedding)
    actions = get_experiments_actions(noise_model, embedding, experiment_id)
    initial_distribution = []
    initial_distribution.append((swap_instance.initial_state[0], 1.00))

    start_time = time.time()
    pomdp = build_pomdp(actions, noise_model, BFS_DISTANCE, embedding, initial_distribution=initial_distribution, guard=guard) # WARNING: 7 is the horizon for which we are interested in this particular experiment for the phaseflip problem
    pomdp.optimize_graph(swap_instance)
    end_time = time.time()
    if return_pomdp:
        return pomdp
    pomdp.serialize(swap_instance, pomdp_write_path)
    return end_time-start_time
    
def load_embeddings(config=None, config_path=None):
    if config is None:
        assert config_path is not None
        config = load_config_file(config_path, RouteExperimentID)
    
    embeddings_path = get_embeddings_path(config)
    
    experiment_id = config["experiment_id"]
    assert isinstance(experiment_id, RouteExperimentID)
    
    with open(embeddings_path, 'r') as file:
        result = dict()
        data = json.load(file)
        result["count"] = data["count"]

        for hardware_spec in HardwareSpec:
            if (hardware_spec.value in config["hardware"]):
                result[hardware_spec] = dict()
                result[hardware_spec]["count"] = data[hardware_spec.value]["count"]
                result[hardware_spec]["embeddings"] = []

                for embedding in data[hardware_spec.value]["embeddings"]:
                    d = dict()
                    for (key, value) in embedding.items():
                        d[int(key)] = int(value)
                    result[hardware_spec]["embeddings"].append(d)
            else:
                assert hardware_spec.value not in data.keys()
        
        return result
    raise Exception(f"could not load embeddings file {POMDP_OUTPUT_DIR}{EMBEDDINGS_FILE}")

def generate_pomdps(config_path):
    config = load_config_file(config_path, RouteExperimentID)
    
    # the file that contains the time to generate the POMDP is in this folder
    output_dir = os.path.join(get_project_path(), config["output_dir"])
    directory_exists(output_dir)
        
     # all pomdps will be outputed in this folder:
    output_folder = os.path.join(output_dir, "pomdps")
    # check that there is a folder with the experiment id inside pomdps path
    directory_exists(output_folder)

    all_embeddings = load_embeddings(config=config)
    
    times_file_path = os.path.join(output_dir, 'pomdp_times.csv')
    times_file = open(times_file_path, "w")
    times_file.write("backend,embedding,time\n")
    for backend in HardwareSpec:
        if backend.value in config["hardware"]:
            # try:
            embeddings = all_embeddings[backend]["embeddings"]
            for (index, m) in enumerate(embeddings):
                print(backend, index, m)
                time_taken = generate_pomdp(RouteExperimentID.EXP1, backend, m, f"{output_folder}/{backend.value}_{index}.txt")
                if time_taken is not None:
                    times_file.write(f"{backend.name},{index},{time_taken}\n")
                times_file.flush()
            # except Exception as err:
            #     print(f"Unexpected {err=}, {type(err)=}")
    times_file.close()
    
    
if __name__ == "__main__":
    arg_backend = sys.argv[1]
    Precision.PRECISION = MAX_PRECISION
    Precision.update_threshold()
    settings = get_project_settings()
    project_path = settings["PROJECT_PATH"]
    
    allowed_hardware = []
    for hardware in HardwareSpec:
        noise_model = NoiseModel(hardware, thermal_relaxation=WITH_THERMALIZATION)
        if noise_model.num_qubits >= 14:
            allowed_hardware.append(hardware)
    if arg_backend == "gen_configs":
        # step 0
        generate_configs(experiment_name="route", experiment_id=RouteExperimentID.EXP1, min_horizon=BFS_DISTANCE, max_horizon=BFS_DISTANCE, allowed_hardware=allowed_hardware)
    elif arg_backend == "embeddings":
        # generate paper embeddings
        batches = get_num_qubits_to_hardware(WITH_THERMALIZATION, allowed_hardware=allowed_hardware)
        for num_qubits in batches.keys():
            exp1_config_path = get_config_path("route", RouteExperimentID.EXP1, num_qubits)
            generate_embeddings(config_path=exp1_config_path, experiment_enum=RouteExperimentID, get_hardware_embeddings=get_hardware_embeddings, experiment_id=RouteExperimentID.EXP1)
    elif arg_backend == "pomdp":
        config_path = sys.argv[2]
        generate_pomdps(config_path)
    elif arg_backend == "all_pomdps":
        # generate paper embeddings
        batches = get_num_qubits_to_hardware(WITH_THERMALIZATION, allowed_hardware=allowed_hardware)
        for num_qubits in batches.keys():
            generate_pomdps(get_config_path("route", RouteExperimentID.EXP1, num_qubits))    

                

        
    