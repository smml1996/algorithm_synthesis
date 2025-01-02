from typing import Dict, List, Set, Tuple
from algorithm import AlgorithmNode
from cmemory import KnwObs
from copy import deepcopy

from pomdp import POMDP, POMDPAction, POMDPVertex
from utils import Queue

class ImperfectGameAlgorithm:
    def __init__(self):
        self.states_to_algorithm = dict()
        self.states_to_comments = dict()
        self.next_states = dict()
        self.initial_state = None
    
    def does_state_exists(self, state):
        return state in self.states_to_algorithm.keys()
    
    def dump(self, path, actions: List[POMDPAction]):
        file = open(path, "w")
        file.write("import os, sys, random\n")
        file.write("sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n")
        file.write("from qiskit import QuantumCircuit, ClassicalRegister\n")
        file.write("from ibm_noise_models import NoiseModel, Instruction, instruction_to_ibm, Op\n\n")
        
        file.write("###### ACTIONS ######\n")
        for action in actions:
            assert isinstance(action, POMDPAction)
            instructions_str = ""
            for instruction in action.instruction_sequence:
                if len(instructions_str) > 0:
                    instructions_str += ", "
                instructions_str += f"Instruction({instruction.target}, Op.{instruction.op.name}, {instruction.control}, {instruction.params})"
            file.write(f"{action.name} = [{instructions_str}]\n")
           
        file.write("def my_algorithm(qc: QuantumCircuit, noise_model: NoiseModel, basis_gates, seed=1):") 
        
        file.write("\t#### INITIALIZE SIMULATOR ######\n")
        file.write("\tsimulator = QSimulator(noise_model, seed)\n")
        file.write(f"\tcurrent_state = {self.initial_state}\n\n")
        
        file.write("\twhile True:\n")
        
        for (state, algo_nodes) in self.states_to_algorithm.items():
            file.write(f"\t\tif current_state == {state}:\n")
            actions_ = [algo_node.action_name for algo_node in algo_nodes] 
            file.write(f"\t\t\tactions = {actions_}\n")
            file.write(f"\t\t\tchoosen_action = random.choice(actions)\n")
            file.write(f"\t\t\tout = simulator.apply_instructions(choosen_action)\n")
            
            if isinstance(self.next_states[state], int):
                file.write(f"\t\t\tcurrent_state = {self.next_states[state]}\n")
            else:
                for (out_val, next_state) in self.next_states[state].items():
                    file.write(f"\t\t\tout == {out_val}:\n")
                    file.write(f"\t\t\t\tcurrent_state == {next_state}:\n")
        
        file.close()
        
class KnwObs:
    vertices: Set[int]
    def __init__(self, vertices, out_val=None):
        self.vertices = frozenset(vertices)
        self.out_val = out_val
    
    def __eq__(self, other_obs):
        assert isinstance(other_obs, KnwObs)
        if self.vertices == other_obs.vertices:
            assert self.out_val == other_obs.out_val
            return True
        return False
        
    def __hash__(self):
        return hash(self.vertices)

class KnwVertex:
    observable: KnwObs
    vertex_id: int
    def __init__(self, observable_vertices: List[POMDPVertex], vertex: POMDPVertex):
        self.observable = KnwObs(observable_vertices)
        self.vertex_id = vertex
        self.state = vertex.classical_state.get_memory_val()
        
    def __eq__(self, other):
        if self.vertex != other.vertex:
            return False
        return self.observable == other.observable
    
    def __hash__(self):
        return hash((self.observable, self.vertex.__hash__()))
    
class KnwGraph:
    ''' Knowledge graph
    '''
    observables_to_vertices: Dict[int, set[int]]
    rankings : Dict[KnwVertex, int]
    transition_matrix: Dict[KnwVertex, Dict[str, Set[KnwVertex]]]
    def __init__(self, pomdp: POMDP, max_depth: int = -1):
        self.rankings = dict()
        self.transition_matrix = dict()
        self.target_vertices = set()
        self.equivalence_class = dict()
        self.actions = pomdp.actions
        self.build_graph(pomdp, max_depth)
            
        
        
    def is_target_vertex(self, vertex, is_target_qs):
        assert isinstance(vertex, KnwVertex)
        for pomdp_state in vertex.observable.vertices:
            if not is_target_qs((pomdp_state.quantum_state, pomdp_state.classical_state)):
                return False
        return True
    
    def build_graph(self, pomdp: POMDP, is_target_qs, max_depth: int=-1):
        assert max_depth != 0
        q = Queue()
        self.initial_state = KnwVertex([pomdp.initial_state], pomdp.initial_state)
        q.push((self.initial_state, 0))
        visited.add(current_vertex)
        
        visited = set()
        
        while not q.is_empty():
            current_vertex, depth = q.pop()
            assert isinstance(current_vertex, KnwVertex)
            
            if max_depth > 0 and depth == max_depth:
                continue
            assert (max_depth == -1) or (not (depth > max_depth))
            current_observable = current_vertex.observable # knowledge
            actual_pomdp_vertex = current_vertex.pomdp_vertex
            
            for action in pomdp.actions:
                assert isinstance(action, POMDPAction)
                
                # compute actual states we could possibly be in
                real_succs_dict = dict()
                for (succ, _prob) in pomdp.transition_matrix[actual_pomdp_vertex][action.name].items():
                    classical_state = succ.classical_state
                    if classical_state not in real_succs_dict.keys():
                        real_succs_dict[classical_state] = set()
                    real_succs_dict[classical_state].add(succ)

                # compute knowledge
                pomdp_obs_to_v = dict()
                for current_v in current_observable.vertices:
                    for (succ, _prob) in pomdp.transition_matrix[current_v][action.name].items():
                        classical_state = succ.classical_state
                        if classical_state not in pomdp_obs_to_v.keys():
                            pomdp_obs_to_v[classical_state] = set()
                        pomdp_obs_to_v[classical_state].add(succ)
                
                # update transition matrix
                for (classical_state, real_succs) in real_succs_dict.items():
                    for real_succ in real_succs:
                        new_vertex = KnwVertex(pomdp_obs_to_v[classical_state], real_succ)
                        if new_vertex.observable not in self.equivalence_class.keys():
                            self.equivalence_class[new_vertex.observable] = set()
                        self.equivalence_class[new_vertex.observable].add(new_vertex)
                        is_target = self.is_target_vertex(new_vertex, is_target_qs)
                        if not (new_vertex in visited):
                            visited.add(new_vertex)
                            if not is_target:
                                # only explore if we are not in a target observable (otherwise it doesnt make sense to keep exploring because we have already reached the target)
                                q.push((new_vertex, depth + 1))
                            else:
                                self.target_vertices.add(new_vertex) # TODO: optimize this (reduce number of checks by caching observables that have already been checked)
                        if action.name not in self.transition_matrix[current_vertex].keys():
                            self.transition_matrix[current_vertex][action.name] = set()
                        assert action.name not in self.transition_matrix[current_vertex].keys()
                        self.transition_matrix[current_vertex][action.name].add(new_vertex)
                    
    def search_pomdp_action(self, delta) -> POMDPAction:
        for action in self.actions:
            if delta == action.name:
                return action
        raise Exception("POMDP Action not found in Knowledge graph")
    
    def get_state_to_algorithm(self) -> ImperfectGameAlgorithm:
        '''returns a dictionary that maps a state (integer) to AlgorithmNode
        '''
        winning_set = find_winning_set(self)
        algorithm = ImperfectGameAlgorithm()
        if self.initial_state not in winning_set:
            return algorithm
        
        q = Queue()
        q.push(self.initial_state)
        
        while not q.is_empty():
            current_vertex = q.pop()
            if algorithm.does_state_exists(current_vertex.state):
                continue
            
            allowed_deltas = clean_deltas(self, class_allow(self, current_vertex, winning_set))
            for delta in allowed_deltas:
                action = self.search_pomdp_action(delta)
                algorithm_node = AlgorithmNode(action.name, action.instruction_sequence)
                algorithm.states_to_algorithm[current_vertex.state] = algorithm_node
                assert current_vertex.state not in algorithm.next_states.keys()
                algorithm.next_states[current_vertex.state] = set()
                for successor in self.transition_matrix[current_vertex][delta]:
                    algorithm.next_states[current_vertex.state].add(successor.state)
        return algorithm

def clean_deltas(graph: KnwGraph, current_vertex, deltas) -> Set[str]:
    new_deltas = set()
    for delta in deltas:
        post_vertices = set()
        assert current_vertex in graph.equivalence_class[current_vertex.observable]
        for vertex in graph.equivalence_class[current_vertex.observable]:
            post_vertices = post_vertices.union(graph.transition_matrix[vertex][delta])
        min_rank = min([graph.rankings[v] for v in post_vertices])
        pre_rank = max(graph.rankings[v] for v in graph.equivalence_class[current_vertex.observable])
        if min_rank < pre_rank:
            if len(graph.adj_list[current_vertex][delta]) != 1 or (current_vertex not in graph.adj_list[current_vertex][delta]):
                new_deltas.add(delta)
    return new_deltas

## FINDING WINNING SET ##
def allow(graph: KnwGraph, v: KnwVertex, Y: Set[KnwVertex]) -> Set[str]:
    result = set()
    for (channel, vertices) in graph.transition_matrix[v].items():
        if vertices.issubset(Y):
            result.add(channel)
    return result
        

def class_allow(graph: KnwGraph, v: KnwVertex, Y: Set[KnwVertex]) -> Set[int]:
    equivalence_class = graph.equivalence_class[v.observable]
    result = allow(graph, v, Y)
    for q in equivalence_class:
        if q != v:
            result = result.intersection(allow(graph, q, Y))
    return result

def apre(graph: KnwGraph, X: Set[KnwVertex], Y: Set[KnwVertex]) -> Set[KnwVertex]:
    result = set()
    for q in Y:
        allow_q = class_allow(graph, q, Y)
        for delta in allow_q:
            post_q = graph.transition_matrix[q][delta]
            if post_q.issubset(X):
                result.add(q)
                break
    return result


def find_winning_set(graph: KnwGraph):
    Y = graph.vertices
    
    Bt = set()
    current_ranking = 0
    for o in graph.target_vertices:
        Bt = Bt.union(o)
        graph.rankings[o] = 0
    
    while True:
        X = set()
        while True:
            X_ = apre(graph, X, Y).union(Bt)

            if X_ == X:
                break
            
            current_ranking += 1
            for x in (X_-X):
                assert x not in graph.rankings.keys()
                graph.rankings[x] = current_ranking
            X = deepcopy(X_)

        Y_ = X
        if Y_ == Y:
            break
        Y = deepcopy(Y_)

    return Y


