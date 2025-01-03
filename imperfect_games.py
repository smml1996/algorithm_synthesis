from typing import Dict, List, Set, Tuple
from algorithm import AlgorithmNode
from copy import deepcopy

from pomdp import POMDP, POMDPAction, POMDPVertex
from utils import Queue

class ImperfectGameAlgorithm:
    def __init__(self):
        self.states_to_algorithm = dict()
        self.states_to_comments = dict()
        self.next_states = dict()
        self.initial_state = None
        self.target_states = set()
    
    def does_state_exists(self, state):
        return state in self.states_to_algorithm.keys()
    
    def dump(self, path, actions: List[POMDPAction]):
        file = open(path, "w")
        file.write("import os, sys, random\n")
        file.write("sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n")
        file.write("from qiskit import QuantumCircuit, ClassicalRegister\n")
        file.write("from simulator import QSimulator\n")
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
           
        file.write("def my_algorithm(qc: QuantumCircuit, noise_model: NoiseModel, seed=1):\n") 
        
        file.write("\t#### INITIALIZE SIMULATOR ######\n")
        file.write("\tsimulator = QSimulator(noise_model, seed)\n")
        file.write(f"\tcurrent_state = {self.initial_state}\n\n")
        
        file.write("\twhile True:\n")
        
        file.write(f"\t\tif current_state in {self.target_states}:\n")
        file.write(f"\t\t\t# target state reached\n")
        file.write(f"\t\t\tbreak\n")

        for (state, algo_nodes) in self.states_to_algorithm.items():
            if state in self.target_states:
                continue
            file.write(f"\t\tif current_state == {state}:\n")
            actions_ = [algo_node.action_name for algo_node in algo_nodes] 
            file.write(f"\t\t\tactions = {actions_}\n")
            file.write(f"\t\t\tchoosen_action = random.choice(actions)\n")
            file.write(f"\t\t\tsimulator.apply_instructions(choosen_action)\n")

            for (delta, classica_states_dict) in self.next_states[state].items():
                for (classical_state, next_state) in classica_states_dict.items():
                    file.write(f"\t\t\tif choosen_action == {delta} and simulator.meas_cache.get_memory_val() == {classical_state.get_memory_val()}:\n")
                    file.write(f"\t\t\t\tcurrent_state = {next_state}\n")
                    file.write(f"\t\t\t\tcontinue\n")
            file.write(f"\t\t\traise Exception('Invalid (classical) memory state at {state}')\n")
        file.close()
        
class KnwObs:
    vertices: Set[int]
    def __init__(self, vertices):
        classical_state = None
        for vertex in vertices:
            if classical_state is None:
                classical_state = vertex.classical_state
            else:
                assert vertex.classical_state == classical_state
        self.vertices = frozenset(vertices)
    
    def __eq__(self, other_obs):
        assert isinstance(other_obs, KnwObs)
        return self.vertices == other_obs.vertices
        
    def __hash__(self):
        return hash(self.vertices)

class KnwVertex:
    observable: KnwObs
    def __init__(self, observable_vertices: List[POMDPVertex], vertex: POMDPVertex):
        self.observable = KnwObs(observable_vertices)
        self.vertex = vertex
        
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
    def __init__(self, pomdp: POMDP, is_target_qs, max_depth: int = -1):
        self.rankings = dict()
        self.transition_matrix = dict()
        self.target_vertices = set()
        self.equivalence_class = dict()
        self.actions = pomdp.actions
        self.vertices = None
        self.initial_states = []
        self.build_graph(pomdp, is_target_qs, max_depth)
            
        
        
    def is_target_vertex(self, vertex, is_target_qs):
        assert isinstance(vertex, KnwVertex)
        for pomdp_state in vertex.observable.vertices:
            if not is_target_qs((pomdp_state.quantum_state, pomdp_state.classical_state)):
                return False
        return True
    
    def build_graph(self, pomdp: POMDP, is_target_qs, max_depth: int=-1):
        assert max_depth != 0
        visited = set()
        q = Queue()
        if "INIT_" in pomdp.transition_matrix[pomdp.initial_state].keys():
            possible_vertices = set()
            for (vertex, prob_) in pomdp.transition_matrix[pomdp.initial_state]["INIT_"].items():
                if prob_ > 0:
                    assert vertex.classical_state.get_memory_val() == 0
                    possible_vertices.add(vertex)
            
            for vertex in possible_vertices:
                knw_vertex = KnwVertex(possible_vertices, vertex)
                self.initial_states.append(knw_vertex)
                visited.add(knw_vertex)
                q.push((knw_vertex, 0))
                
        else:
            self.initial_states = [KnwVertex([pomdp.initial_state], pomdp.initial_state)]
            # model initial distribution
            q.push((self.initial_state, 0))
            visited.add(self.initial_state)
        
        
        while not q.is_empty():
            current_vertex, depth = q.pop()
            assert isinstance(current_vertex, KnwVertex)
            
            if max_depth > 0 and depth == max_depth:
                continue
            assert (max_depth == -1) or (not (depth > max_depth))
            current_observable = current_vertex.observable # knowledge
            actual_pomdp_vertex = current_vertex.vertex
            
            for action in pomdp.actions:
                assert isinstance(action, POMDPAction)
                
                # compute actual states we could possibly be in
                real_succs_dict = dict()
                for (succ, prob) in pomdp.transition_matrix[actual_pomdp_vertex][action.name].items():
                    if prob > 0:
                        classical_state = succ.classical_state
                        if classical_state not in real_succs_dict.keys():
                            real_succs_dict[classical_state] = set()
                        real_succs_dict[classical_state].add(succ)

                # compute knowledge
                pomdp_obs_to_v = dict()
                for current_v in current_observable.vertices:
                    for (succ, prob) in pomdp.transition_matrix[current_v][action.name].items():
                        if prob > 0:
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
                        if current_vertex not in self.transition_matrix.keys():
                            self.transition_matrix[current_vertex] = dict()
                        if action.name not in self.transition_matrix[current_vertex].keys():
                            self.transition_matrix[current_vertex][action.name] = set()
                        self.transition_matrix[current_vertex][action.name].add(new_vertex)
        self.vertices = visited
                    
    def search_pomdp_action(self, delta) -> POMDPAction:
        for action in self.actions:
            if delta == action.name:
                return action
        raise Exception("POMDP Action not found in Knowledge graph")
    
    def get_algorithm(self) -> ImperfectGameAlgorithm:
        '''returns a dictionary that maps a state (integer) to AlgorithmNode
        '''
        winning_set = find_winning_set(self)
        
        assert self.initial_states is not None
        # assign to each diff. observable a state (int)
        obs_to_indices = dict()
        count_obs = 0
        for vertex in self.rankings.keys():
            assert isinstance(vertex, KnwVertex)
            obs = vertex.observable
            if obs not in obs_to_indices.keys():
                obs_to_indices[obs] = count_obs
                count_obs += 1
                
        algorithm = ImperfectGameAlgorithm()
        algorithm.initial_state = obs_to_indices[self.initial_states[0].observable]
        q = Queue()
        visited = set()
        for initial_state in self.initial_states:
            q.push(initial_state)
            visited.add(initial_state)
            if initial_state not in winning_set:
                return algorithm
        
        
        while not q.is_empty():
            current_vertex = q.pop()
            current_state = obs_to_indices[current_vertex.observable]
            if current_state not in algorithm.states_to_algorithm.keys():
                algorithm.states_to_algorithm[current_state] = []
            
            if current_state not in algorithm.next_states.keys():
                algorithm.next_states[current_state] = dict()

            allowed_deltas = clean_deltas(self, current_vertex, class_allow(self, current_vertex, winning_set))
            for delta in allowed_deltas:
                action = self.search_pomdp_action(delta)
                algorithm_node = AlgorithmNode(action.name, action.instruction_sequence)
                if algorithm_node not in algorithm.states_to_algorithm[current_state]:
                    algorithm.states_to_algorithm[current_state].append(algorithm_node)
                
                if delta not in algorithm.next_states[current_state].keys():
                    algorithm.next_states[current_state][delta] = dict()
                
                for successor in self.transition_matrix[current_vertex][delta]:
                    if successor not in visited:
                        q.push(successor)
                        visited.add(successor)
                    if successor.vertex.classical_state in algorithm.next_states[current_state][delta].keys():
                        assert algorithm.next_states[current_state][delta][successor.vertex.classical_state] == obs_to_indices[successor.observable]
                    else: 
                        algorithm.next_states[current_state][delta][successor.vertex.classical_state] = obs_to_indices[successor.observable]
        
        for target_state in self.target_vertices:
            algorithm.target_states.add(obs_to_indices[target_state.observable])
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
            if len(graph.transition_matrix[current_vertex][delta]) != 1 or (current_vertex not in graph.transition_matrix[current_vertex][delta]):
                new_deltas.add(delta)
    return new_deltas

## FINDING WINNING SET ##
def allow(graph: KnwGraph, v: KnwVertex, Y: Set[KnwVertex]) -> Set[str]:
    result = set()
    if v not in graph.transition_matrix.keys():
        return result
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
    
    Bt = graph.target_vertices
    current_ranking = 0
    for o in graph.target_vertices:
        graph.rankings[o] = 0
    
    while True:
        X = set()
        while True:
            X_ = apre(graph, X, Y).union(Bt)

            if X_ == X:
                break
            
            current_ranking += 1
            for x in (X_-X):
                if x not in graph.rankings.keys():
                    graph.rankings[x] = current_ranking
            X = deepcopy(X_)

        Y_ = X
        if Y_ == Y:
            break
        Y = deepcopy(Y_)

    return Y


