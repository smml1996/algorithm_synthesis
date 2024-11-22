from cmath import isclose
import os, sys
sys.path.append(os.getcwd()+"/..")
from typing import Dict, List, Tuple
import json

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from algorithm import AlgorithmNode, execute_algorithm
from pomdp import POMDPAction
from qpu_utils import Op
from utils import Precision, Queue, are_matrices_equal

from qstates import QuantumState
from ibm_noise_models import Instruction, NoiseModel, get_ibm_noise_model, HardwareSpec, ibm_simulate_circuit, load_config_file
from enum import Enum
from experiments_utils import ReadoutNoise, directory_exists, get_config_path, get_default_algorithm, get_embedding_index,get_guarantees, get_num_qubits_to_hardware, get_project_path

from bitflip import get_hardware_embeddings as get_hardware_embeddings_z
from ghz import get_hardware_embeddings as get_hardware_embeddings_ghz

from ghz import GHZExperimentID, GHZInstance
from bitflip import BitflipExperimentID
from phaseflip import PhaseflipExperimentID
import bitflip
import phaseflip
import ghz
from ibm_noise_models import NoiselessCCZ, NoiselessCCX, NoiselessX, NoiselessCX

WITH_TERMALIZATION = False
MAX_PRECISION = 10
LIMIT_EMBEDDINGS = 10
BFS_DISTANCE = 3

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

def get_shortest_distance(noise_model: NoiseModel, qubit1:int, qubit2: int):
    distance = dict()
    
    q = Queue()
    visited = set()
    q.push(qubit1)
    visited.add(qubit1)
    distance[qubit1] = 0
    
    g = noise_model.digraph
    while not q.is_empty():
        current_qubit = q.pop()
        current_distance = distance[current_qubit]
        for succ in g[succ]:
            if succ not in distance.keys():
                distance[succ] = BFS_DISTANCE+1
            distance[succ] = min(distance[succ], current_distance + 1)
            if succ not in visited:
                if current_distance < BFS_DISTANCE:
                    visited.add(succ)
                    q.push(succ)
    
                
    if qubit2 in distance.keys():
        return distance[qubit2]
    return BFS_DISTANCE + 1

def is_intersection_empty(l1, l2) -> bool:
    a = set()
    b = set()
    for l in l1:
        a.add(l)
        
    for l in l2:
        b.add(l)
    return a.isdisjoint(b)

def get_routing(routes1: List[List[int]], routes2: List[List[int]]) -> Tuple[List[int], List[int]]:

    for route1 in routes1:
        for route2 in routes2:
            if is_intersection_empty(route1, route2):
                return route1, route2
    return None, None
            
def get_routing_algoritm(route, embedding, index=0, rev=False) -> AlgorithmNode:
    if not rev:
        if index >= len(route)-1:
            return None
    else:
        if index < 0:
            return None
    
    if rev:
        p_control = route[index]
        p_target = route[index+1]
    else:
        p_control = route[index-1]
        p_target = route[index]
    v_control = embedding[p_control]
    v_target = embedding[p_target]
    
    head = AlgorithmNode(f"CX{v_control}_{v_target}", [
        Instruction(v_target, Op.CNOT, control=v_control)
    ])
    
    if not rev:
        head.next_ins = get_routing_algoritm(route, index=index+1)
        
    else:
        head.next_ins = get_routing_algoritm(route, index=index-1)
    return head

    
def does_routing_exists(routes1, routes2):
    alg1, alg2 = get_routing(routes1, routes2)
    return (alg1 is not None) and (alg2 is not None)

def is_feasible(noise_model, qubit1, qubit2, tqubit1, tqubit2):
    possible_routes11 = get_possible_routes(noise_model.digraph, qubit1, tqubit1, 0)
    possible_routes22 = get_possible_routes(noise_model.digraph, qubit2, tqubit2, 0)
    if len(possible_routes11) > 0 and len(possible_routes22) > 0:
        if does_routing_exists(possible_routes11, possible_routes22):
            return True
    
    possible_routes12 = get_possible_routes(noise_model.digraph, qubit1, tqubit2, 0)
    possible_routes21 = get_possible_routes(noise_model.digraph, qubit2, tqubit1, 0)
    if len(possible_routes12) > 0 and len(possible_routes21) > 0:
        if does_routing_exists(possible_routes12, possible_routes21):
            return True
        
    return False

def merge_algorithms(alg1, alg2):
    if alg1 is None:
        raise Exception("I was not expecting this")
    if alg1.next_ins is None:
        alg1.next_ins = alg2
    else:
        merge_algorithms(alg1.next_ins, alg2)

def get_final_routing_algorithm(route1, route2, embedding):
    alg1 = get_routing_algoritm(route1, embedding)
    alg2 = get_routing_algoritm(route2, embedding)
    alg = merge_algorithms(alg1, alg2)
    alg_dagger = merge_algorithms(
        get_routing_algoritm(route2, embedding, index=len(route2)-2, rev=True),
        get_routing_algoritm(route1, embedding, index=len(route1)-2, rev=True))
    return alg, alg_dagger

def find_routings(noise_model, qubit1, qubit2, tqubit1, tqubit2):
    possible_routes11 = get_possible_routes(noise_model.digraph, qubit1, tqubit1, 0)
    possible_routes22 = get_possible_routes(noise_model.digraph, qubit2, tqubit2, 0)
    route1, route2 = get_routing(possible_routes11, possible_routes22)
    if not (route1 is None) and not (route2 is None):
        return route1, route2
    
    possible_routes12 = get_possible_routes(noise_model.digraph, qubit1, tqubit2, 0)
    possible_routes21 = get_possible_routes(noise_model.digraph, qubit2, tqubit1, 0)
    route1, route2 = get_routing(possible_routes12, possible_routes21)
    if not (route1 is None) and not (route2 is None):
        return route1, route2
        
    return None, None


def get_embedding(ghz_embedding, meas12_embedding, meas13_embedding) -> Dict[int, int]:
    return {
        'ghz': ghz_embedding,
        'meas12': meas12_embedding,
        'meas13': meas13_embedding
    }
    
def get_required_qubits(embedding):
    s = set()
    for (name, e) in embedding.items():
        for v in e.values():
            s.add(v)
    return len(s)

def get_possible_routes(graph, current_qubit, target_qubit, current_depth)-> List[int]:
    assert current_depth <= BFS_DISTANCE
    assert isinstance(target_qubit, int)
    assert isinstance(current_qubit, int) 
    if current_qubit == target_qubit:
            return [[target_qubit]]
    if current_depth == BFS_DISTANCE:
        return []
    possible_routes_ = []
    for succ in graph[current_qubit]:
        assert isinstance(succ, int)
        for possible_route_ in get_possible_routes(graph, succ, target_qubit, current_depth+1):
            possible_routes_.append(possible_route_)
    answer = []
    for possible_route in possible_routes_:
        if len(possible_route) > 0:
            temp = [current_qubit]
            
            for v in possible_route:
                temp.append(v)
            answer.append(temp)
    return answer

  
def get_hardware_embeddings(hardware_spec):
    answer = []
    ghz_embeddings = get_hardware_embeddings_ghz(hardware_spec, GHZExperimentID.EXP1)
    embeddings_bitflip = get_hardware_embeddings_z(hardware_spec, experiment_id=BitflipExperimentID.IPMA2)
    
    for ghz_embedding in ghz_embeddings:
        qubit1 = ghz_embedding[0]
        qubit2 = ghz_embedding[1]
        qubit3 = ghz_embedding[2]
        for bf12_embedding in embeddings_bitflip:
            if bf12_embedding[2] not in ghz_embedding.values(): # we dont want to measure a qubit that stores the ghz state
                if is_feasible(noise_model, qubit1, qubit2, bf12_embedding[0], bf12_embedding[1]):
                    for bf13_embedding in embeddings_bitflip:
                        if bf13_embedding[2] not in ghz_embedding.values():
                            if is_feasible(noise_model, qubit1, qubit3, bf13_embedding[0], bf13_embedding[1]):
                                temp = get_embedding(ghz_embedding, bf12_embedding, bf13_embedding)
                                if get_required_qubits(temp) < 15:
                                    answer.append(temp) 
    return answer[:LIMIT_EMBEDDINGS]

def get_horizon(experiment_id) -> int:
    if isinstance(experiment_id, GHZExperimentID):
        return 3
    return 7


def get_actions(noise_model, embedding, experiment_id) -> List[POMDPAction]:
    if isinstance(experiment_id, GHZExperimentID):
        actions = []
        assert len(embedding.keys()) == 3
        for control in range(3):
            assert embedding[control] == control
            actions.append(POMDPAction(f"H{control}", [Instruction(control, Op.H)]))
            for target in range(3):
                if control != target:
                    actions.append(POMDPAction(f"CX{control}{target}", [Instruction(target, Op.CNOT, control=control)]))
        return actions
    if isinstance(experiment_id, BitflipExperimentID):
        return bitflip.get_experiments_actions(noise_model, embedding, experiment_id)
    assert isinstance(experiment_id, PhaseflipExperimentID)
    return phaseflip.get_experiments_actions(noise_model, embedding, experiment_id)

def is_only_detect(algorithm_node: AlgorithmNode, actions_to_instructions) -> bool:
    if algorithm_node is None:
        return True
    
    if algorithm_node.instruction_sequence == actions_to_instructions["Z0"]:
            return False
    if algorithm_node.instruction_sequence == actions_to_instructions["X0"]:
        return False
    
    return is_only_detect(algorithm_node.next_ins, actions_to_instructions) and is_only_detect(algorithm_node.case0, actions_to_instructions) and is_only_detect(algorithm_node.case1, actions_to_instructions)

def modify_to_detect(algorithm_node: AlgorithmNode, target_qubit, actions_to_instructions, is_prev_meas=False, found_flip=False) -> None:
    assert isinstance(target_qubit, int)
    if not (algorithm_node is None):
        if algorithm_node.has_meas_instruction():
            algorithm_node.case0 = modify_to_detect(algorithm_node.case0, target_qubit, actions_to_instructions, is_prev_meas=True, found_flip=found_flip)
            algorithm_node.case1 = modify_to_detect(algorithm_node.case1, target_qubit, actions_to_instructions, is_prev_meas=True, found_flip=found_flip)
            return algorithm_node
        elif algorithm_node.instruction_sequence == actions_to_instructions["Z0"] or algorithm_node.instruction_sequence == actions_to_instructions["X0"]:
            new_node = AlgorithmNode(algorithm_node.action_name, [Instruction(target_qubit, Op.X)], noiseless=True)
            new_node.next_ins = modify_to_detect(algorithm_node.next_ins, target_qubit, actions_to_instructions, is_prev_meas=False, found_flip=True)
            return new_node
        else:
            algorithm_node.next_ins = modify_to_detect(algorithm_node.next_ins, target_qubit, actions_to_instructions, is_prev_meas=False, found_flip=found_flip)
            return algorithm_node
    else:
        if is_prev_meas:
            assert not found_flip
            return None
        else:
            assert found_flip
            return algorithm_node
    
def get_actions_to_instructions(noise_model, embedding, experiment_id):
    actions_to_instructions = dict()
    actions = get_actions(noise_model, embedding, experiment_id)
    for action in actions:
        actions_to_instructions[action.name] = action.instruction_sequence
    actions_to_instructions["halt"] = []
    return actions_to_instructions

def load_algorithm(batch, hardware_spec, noise_model, embedding_index, experiment_id, embedding, target_qubit=None):
    horizon = get_horizon(experiment_id)
    experiment_obj = type(experiment_id)
    config_path = get_config_path(experiment_id, batch)
    config = load_config_file(config_path, experiment_obj)
    output_dir = os.path.join(get_project_path(), config["output_dir"])
    algorithms_path = os.path.join(output_dir, "algorithms")
    algorithm_path = os.path.join(algorithms_path, f"{hardware_spec.value}_{embedding_index}_{horizon}.json")
    
    actions_to_instructions = get_actions_to_instructions(noise_model, embedding, experiment_id)
    if "Z0" not in actions_to_instructions.keys():
        actions_to_instructions["Z0"] = None
    if "X0" not in actions_to_instructions.keys():
        actions_to_instructions["X0"] = None
    # load algorithm json
    f_algorithm = open(algorithm_path)
    algorithm = AlgorithmNode(serialized=json.load(f_algorithm), actions_to_instructions=actions_to_instructions)
    f_algorithm.close()  
    if not isinstance(experiment_id, GHZExperimentID):
        assert target_qubit is not None
        algorithm = modify_to_detect(algorithm, target_qubit, actions_to_instructions)
        assert is_only_detect(algorithm, actions_to_instructions)
    return algorithm

class AlgorithmGluer:
    def get_syndrome_qubits(self, noise_model):
        answer = []
        for i in range(noise_model.num_qubits):
            if not (i in self.used_qubits.keys()):
                answer.append(i)
                if len(answer) == 4:
                    return answer
        raise Exception(f"Not enough qubits to run {self.hardware_spec}")
    
    def __init__(self, noise_model, embeddings, batch, hardware_spec, embedding_indices, is_default=False) -> None:
        self.counter_vars = 0
        self.all_embeddings = embeddings        
        self.final_embedding = dict()
        self.used_qubits = dict()
        self.counter = 0
        self.address_space = dict()
        self.address_space["ghz"] = dict()
        self.address_space["meas12"] = dict()
        self.address_space["meas13"] = dict()
        
        for name in ["ghz", "meas12", "meas13"]:
            for (key, value) in self.all_embeddings[name].items():
                if value in self.used_qubits.keys():
                    final_key = self.used_qubits[value]
                else:
                    final_key = self.counter
                    self.final_embedding[final_key] = value
                    self.counter+=1
                    self.used_qubits[value] = final_key
                assert key not in self.address_space[name]
                self.address_space[name][key] = final_key
                
        syndrome_qubits = self.get_syndrome_qubits(noise_model)
        assert len(syndrome_qubits) == 4
        
        for qubit in syndrome_qubits:
            assert qubit not in self.used_qubits.keys()
            final_key = self.counter
            self.counter += 1
            self.final_embedding[final_key] = qubit
            self.used_qubits[qubit] = final_key

            
        self.qubitbf12 = self.used_qubits[syndrome_qubits[0]]
        self.qubitbf13 = self.used_qubits[syndrome_qubits[1]]
        self.qubitpf12 = self.used_qubits[syndrome_qubits[2]]
        self.qubitpf13 = self.used_qubits[syndrome_qubits[3]]
        if is_default:
            self.ghz = get_default_algorithm(noise_model, self.address_space["ghz"], GHZExperimentID.EXP1)
            if self.ghz is None:
                self.ghz = load_algorithm(batch, hardware_spec, noise_model, embedding_indices["ghz"], GHZExperimentID.EXP1, self.address_space["ghz"])
            self.bf12 = get_default_algorithm(noise_model, self.address_space["bf_meas12"], BitflipExperimentID.IPMA2, target_qubit=self.qubitbf12)
            self.bf13 = get_default_algorithm(noise_model, self.address_space["bf_meas13"], BitflipExperimentID.IPMA2, target_qubit=self.qubitbf13)
            self.pf12 = get_default_algorithm(noise_model, self.address_space["pf_meas12"], PhaseflipExperimentID.IPMA, target_qubit=self.qubitpf12)
            self.pf13 = get_default_algorithm(noise_model, self.address_space["pf_meas13"], PhaseflipExperimentID.IPMA, target_qubit=self.qubitpf13)
        else:
            self.ghz = load_algorithm(batch, hardware_spec, noise_model, embedding_indices["ghz"], GHZExperimentID.EXP1, self.address_space["ghz"])
            self.bf12 = load_algorithm(batch, hardware_spec, noise_model, embedding_indices["bf_meas12"], BitflipExperimentID.IPMA2, self.address_space["meas12"], target_qubit=self.qubitbf12)
            self.bf13 = load_algorithm(batch, hardware_spec, noise_model, embedding_indices["bf_meas13"], BitflipExperimentID.IPMA2, self.address_space["meas13"], target_qubit=self.qubitbf13)
            self.pf12 = load_algorithm(batch, hardware_spec, noise_model, embedding_indices["pf_meas12"], PhaseflipExperimentID.IPMA, self.address_space["meas12"], target_qubit=self.qubitpf12)
            self.pf13 = load_algorithm(batch, hardware_spec, noise_model, embedding_indices["pf_meas13"], PhaseflipExperimentID.IPMA, self.address_space["meas13"], target_qubit=self.qubitpf13)
            
    @property
    def num_qubits(self):
        return len(self.final_embedding.keys())
    
    def is_target_qs(self, state_vector):
        qs = None
        for (index, amp) in enumerate(state_vector):
            if not isclose(amp, 0.0, abs_tol=Precision.isclose_abstol):
                if qs is None:
                    qs = QuantumState(index, amp, qubits_used=list(self.final_embedding.keys()))
                else:
                    assert qs.get_amplitude(index) == 0.0
                    qs.insert_amplitude(index, amp) 
        remove_indices = [x for x in range(3, len(self.final_embedding.keys()))]
        rho = qs.multi_partial_trace(remove_indices=remove_indices)
        assert len(rho) == 8
        assert len(rho[0]) == 8
        
        for i in range(len(rho)):
            for j in range(len(rho)):
                if (i == 0 and j ==0) or (i==0 and j == 7) or (i==7 and j ==0) or (i == 7 and j == 7):
                    if not isclose(rho[i][j], 1.0, abs_tol=Precision.isclose_abstol):
                        return False
                else:
                    if not isclose(rho[i][j], 0.0, abs_tol=Precision.isclose_abstol):
                        return False
        return True
    
    def algorithm(self, qubit, error, qc, cbits):
        # ghz algorithm
        execute_algorithm(self.ghz, qc, cbits=cbits)
        
        # insert error
        new_virtual_address = self.address_space["ghz"][qubit]
        if error == Op.X:
            qc.x(new_virtual_address)
        elif error == Op.Z:
            qc.z(new_virtual_address)
        elif error == Op.Y:
            qc.x(new_virtual_address)
            qc.z(new_virtual_address)
            
        qubit0 = self.address_space["ghz"][0]
        qubit1 = self.address_space["ghz"][1]
        qubit2 = self.address_space["ghz"][2]
        
            
        # bitflip measurement 1-2
        bf12_0 = self.address_space["meas12"][0]
        bf12_1 = self.address_space["meas12"][1]
        assert bf12_0 != bf12_1
        if qubit0 != bf12_0:
            qc.append(NoiselessCX, [qubit0, bf12_0])
        if qubit1 != bf12_1:
            qc.append(NoiselessCX, [qubit1, bf12_1])
        execute_algorithm(self.bf12, qc, cbits=cbits)
        if qubit0 != bf12_0:
            qc.append(NoiselessCX, [qubit0, bf12_0])
        if qubit1 != bf12_1:
            qc.append(NoiselessCX, [qubit1, bf12_1])
        
        # bitflip measurement 1-3
        bf13_0 = self.address_space["meas13"][0]
        bf13_1 = self.address_space["meas13"][1]
        assert bf13_0 != bf13_1
        if qubit0 != bf13_0:
            qc.append(NoiselessCX, [qubit0, bf13_0])
        if qubit2 != bf13_1:
            qc.append(NoiselessCX, [qubit2, bf13_1])
        execute_algorithm(self.bf13, qc, cbits=cbits)
        if qubit0 != bf13_0:
            qc.append(NoiselessCX, [qubit0, bf13_0])
        if qubit2 != bf13_1:
            qc.append(NoiselessCX, [qubit2, bf13_1])
        
        ## correcting bitflips
        qc.append(NoiselessCCX, [self.qubitbf12, self.qubitbf13, self.address_space["ghz"][0]])
        
        qc.append(NoiselessX, [self.qubitbf13])
        qc.append(NoiselessCCX, [self.qubitbf12, self.qubitbf13, self.address_space["ghz"][1]])
        qc.append(NoiselessX, [self.qubitbf13])
        
        qc.append(NoiselessX, [self.qubitbf12])
        qc.append(NoiselessCCX, [self.qubitbf12, self.qubitbf13, self.address_space["ghz"][1]])
        qc.append(NoiselessX, [self.qubitbf12])
        
        # phaseflip measurement 1-2
        if qubit0 != bf12_0:
            qc.append(NoiselessCX, [qubit0, bf12_0])
        if qubit1 != bf12_1:
            qc.append(NoiselessCX, [qubit1, bf12_1])
        execute_algorithm(self.pf12, qc, cbits=cbits)
        if qubit0 != bf12_0:
            qc.append(NoiselessCX, [qubit0, bf12_0])
        if qubit1 != bf12_1:
            qc.append(NoiselessCX, [qubit1, bf12_1])
        
        # phaseflip measurement 1-3
        if qubit0 != bf13_0:
            qc.append(NoiselessCX, [qubit0, bf13_0])
        if qubit2 != bf13_1:
            qc.append(NoiselessCX, [qubit2, bf13_1])
        execute_algorithm(self.pf13, qc, cbits=cbits)
        if qubit0 != bf13_0:
            qc.append(NoiselessCX, [qubit0, bf13_0])
        if qubit2 != bf13_1:
            qc.append(NoiselessCX, [qubit2, bf13_1])
        
        ## correcting phaseflips
        qc.append(NoiselessCCZ, [self.qubitpf12, self.qubitpf13, self.address_space["ghz"][0]])
        
        qc.append(NoiselessX, [self.qubitpf13])
        qc.append(NoiselessCCZ, [self.qubitpf12, self.qubitpf13, self.address_space["ghz"][1]])
        qc.append(NoiselessX, [self.qubitpf13])
        
        qc.append(NoiselessX, [self.qubitpf12])
        qc.append(NoiselessCCZ, [self.qubitpf12, self.qubitpf13, self.address_space["ghz"][1]])
        qc.append(NoiselessX, [self.qubitpf12])
        
        

def ibm_simulate_algorithms(noise_model, embeddings, batch, hardware_spec, embedding_indices, is_default) -> float:
    procedure = AlgorithmGluer(noise_model, embeddings, batch, hardware_spec, embedding_indices, is_default=is_default)
    
    ibm_noise_model = get_ibm_noise_model(hardware_spec, thermal_relaxation=WITH_TERMALIZATION)
    
    qr = QuantumRegister(procedure.num_qubits)
    cr = ClassicalRegister(procedure.num_qubits)
    
    qc = QuantumCircuit(qr, cr)
    initial_layout = dict()
    for (key, val) in procedure.final_embedding.items():
        initial_layout[qr[key]] = val
    for qubit in range (0, 3):
        for error in [Op.X, Op.Z, Op.Y, Op.I]:
            procedure.algorithm(qubit, error, qc, cbits=cr)
            qc.save_statevector('res', pershot=True)
            state_vs = ibm_simulate_circuit(qc, ibm_noise_model, initial_layout)
            for (index,state) in enumerate(state_vs):
                print(f"{index+1}/{len(state_vs)}")
                if procedure.is_target_qs(state):
                    accuracy += 1
    
    return 0.0

def get_experiment_id(name):
    if name == "ghz":
        return GHZExperimentID.EXP1
    if name == "bf_meas12" or name == "bf_meas13":
        return BitflipExperimentID.IPMA2
    
    assert name in ["pf_meas12", "pf_meas13"]
    return PhaseflipExperimentID.IPMA
    
if __name__ == "__main__":
    Precision.PRECISION = MAX_PRECISION
    Precision.update_threshold()
    # we only compute on these hardware
    allowed_hardware = []
    for hardware in HardwareSpec:
        noise_model = NoiseModel(hardware, thermal_relaxation=WITH_TERMALIZATION)
        if noise_model.num_qubits >= 14:
            allowed_hardware.append(hardware)
    batches = get_num_qubits_to_hardware(WITH_TERMALIZATION, allowed_hardware=allowed_hardware)
    
    # set up working directory
    project_path = get_project_path()
    output_dir = os.path.join(project_path, "results", "3q_code")
    directory_exists(output_dir)
    # init results csv
    output_file_path = os.path.join(output_dir, "results.csv")
    output_file = open(output_file_path, "w")
    columns = ["hardware_spec"
               "ghz_e_index",
                "my_ghz",
               "default_ghz",
               "bf12_e_index",
                "my_bitflip12",
               "default_bitflip12",
               "bf13_e_index",
               "my_bitflip13",
               "default_bitflip13",
               "pf12_e_index",
               "my_phaseflip12",
               "default_phaseflip12",
               "pf12_e_index",
               "my_phaseflip13",
               "default_phaseflip13",
               "my_acc",
               "default_acc",
               "diff"
               ]
    output_file.write(",".join(columns) + "\n")
    
    
    # rows computations begins here
    for (num_qubits, hardware_specs) in batches.items():
        for hardware_spec in hardware_specs:
            noise_model = NoiseModel(hardware_spec, thermal_relaxation=WITH_TERMALIZATION)
            embeddings = get_hardware_embeddings(hardware_spec)
            for embedding in embeddings:
                my_ghz, default_ghz = get_guarantees(noise_model, num_qubits, hardware_spec, embedding["ghz"], GHZExperimentID.EXP1, 3, ghz.get_experiments_actions, ghz.get_hardware_embeddings, IBMInstanceObj=ghz.IBMGHZInstance)
                embedding_indices = dict()
                for (name1, name2) in [("ghz", "ghz"), ("bf_meas12","meas12"), ("bf_meas13","meas13"), ("pf_meas12", "meas12"), ("pf_meas13", "meas13")]:
                    if name1 == "ghz":
                        embedding_indices[name1] = get_embedding_index(hardware_spec, embedding[name2], get_experiment_id(name1), ghz.get_hardware_embeddings)
                    elif name1[:2] == "bf":
                        embedding_indices[name1] = get_embedding_index(hardware_spec, embedding[name2], get_experiment_id(name1), bitflip.get_hardware_embeddings)
                    else:
                        assert name1[:2] == "pf"
                        embedding_indices[name1] = get_embedding_index(hardware_spec, embedding[name2], get_experiment_id(name1), phaseflip.get_hardware_embeddings)
                ghz_e_index = embedding_indices["ghz"]
                bf12_e_index = embedding_indices["bf_meas12"]
                bf13_e_index = embedding_indices["bf_meas13"]
                pf12_e_index = embedding_indices["pf_meas12"]
                pf13_e_index = embedding_indices["pf_meas13"]
                    
                
                my_bitflip12, default_bitflip12 = get_guarantees(noise_model, num_qubits, hardware_spec, embedding["meas12"], BitflipExperimentID.IPMA2, 7, bitflip.get_experiments_actions, bitflip.get_hardware_embeddings)
                
                my_phaseflip12, default_phaseflip12 = get_guarantees(noise_model, num_qubits, hardware_spec, embedding["meas12"], PhaseflipExperimentID.IPMA, 7, phaseflip.get_experiments_actions, phaseflip.get_hardware_embeddings)
                
                my_bitflip13, default_bitflip13, = get_guarantees(noise_model, num_qubits, hardware_spec, embedding["meas13"], BitflipExperimentID.IPMA2, 7, bitflip.get_experiments_actions, bitflip.get_hardware_embeddings)
                
                my_phaseflip13, default_phaseflip13, = get_guarantees(noise_model, num_qubits, hardware_spec, embedding["meas13"], PhaseflipExperimentID.IPMA, 7, phaseflip.get_experiments_actions, phaseflip.get_hardware_embeddings)
                
                my_acc = ibm_simulate_algorithms(noise_model, embedding, num_qubits, hardware_spec, embedding_indices, is_default=False)
                default_acc = ibm_simulate_algorithms(noise_model, embedding, num_qubits, hardware_spec, embedding_indices, is_default=True)
                diff = my_acc - default_acc
                
                line_elements = [
                hardware_spec.value,
                ghz_e_index,
                my_ghz,
                default_ghz,
                bf12_e_index,
                my_bitflip12,
                default_bitflip12,
                bf13_e_index,
                my_bitflip13,
                default_bitflip13,
                pf12_e_index,
                my_phaseflip12,
                default_phaseflip12,
                pf13_e_index,
                my_phaseflip13,
                default_phaseflip13,
                my_acc,
                default_acc,
                diff
                ]
                for i in range(len(line_elements)):
                    line_elements[i] = str(line_elements[i])
                
                output_file.write(",".join(line_elements) + "\n")
            output_file.flush()
            
    output_file.close()
        