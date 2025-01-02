from typing import Dict, List, Set, Tuple
from cmemory import KnwObs
from copy import deepcopy

from pomdp import POMDP, POMDPAction, POMDPVertex
from utils import Queue

class KnwObs:
    vertices: Set[int]
    def __init__(self, vertices):
        self.vertices = set()
        for v in vertices:
            self.vertices.add(v)
    
    def __eq__(self, other_obs):
        assert isinstance(other_obs, KnwObs)
        return self.vertices == other_obs.vertices
        
    def __hash__(self):
        return hash(tuple(self.vertices))

class KnwVertex:
    observable: KnwObs
    vertex_id: int
    def __init__(self, observable_vertices: List[POMDPVertex], vertex: POMDPVertex):
        self.observable = KnwObs(observable_vertices)
        self.vertex_id = vertex
        
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
    rankings : Dict[int, int]
    transition_matrix: Dict[KnwVertex, Dict[str, KnwVertex]]
    def __init__(self, pomdp: POMDP, max_depth: int = -1):
        self.rankings = dict()
        self.transition_matrix = dict()
        self.actions = [a.name for a in pomdp.actions]
        self.build_graph(pomdp, max_depth)
        self.target_vertices = set()
        
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
                real_successors = action.get_real_successor(actual_pomdp_vertex) # return dict classical_state -> POMDPVertex
                
                # compute knowledge
                pomdp_obs_to_v = dict()
                for current_v in current_observable.vertices:
                    for (succ, _prob) in pomdp.transition_matrix[current_v][action.name].items():
                        classical_state = succ.classical_state
                        if classical_state not in pomdp_obs_to_v.keys():
                            pomdp_obs_to_v[classical_state] = set()
                        pomdp_obs_to_v[classical_state].add(succ)
                
                assert len(pomdp_obs_to_v.keys()) == len(real_successors.keys())
                
                # update transition matrix
                for (classical_state, real_succ) in real_successors.items():
                    new_vertex = KnwVertex(pomdp_obs_to_v[classical_state], real_succ)
                    is_target = self.is_target_vertex(new_vertex, is_target_qs)
                    if not (new_vertex in visited):
                        visited.add(new_vertex)
                        if not is_target:
                            # only explore if we are not in a target observable (otherwise it doesnt make sense to keep exploring because we have already reached the target)
                            q.push((new_vertex, depth + 1))
                        else:
                            self.target_vertices.add(new_vertex) # TODO: optimize this (reduce number of checks by caching observables that have already been checked)
                    assert action.name not in self.transition_matrix[current_vertex].keys()
                    self.transition_matrix[current_vertex][action.name] = new_vertex


## FINDING WINNING SET ##
def allow(graph: KnwGraph, v: KnwVertex, Y: Set[KnwVertex]) -> Set[str]:
    result = set()
    for (channel, vertices_dict) in graph.transition_matrix[v].items():
        vertices = set(vertices_dict.keys())
        if vertices.issubset(Y):
            result.add(channel)
    return result
        

def class_allow(graph: KnwGraph, v: KnwVertex, Y: Set[KnwVertex]) -> Set[int]:
    observable = v.classical_state
    equivalence_class = deepcopy(graph.get_observable_vertices(observable))
    result = allow(graph, v, Y)
    equivalence_class.remove(v)
    for q in equivalence_class:
        result = result.intersection(allow(graph, q, Y))
    return result

def apre(graph: KnwGraph, X: Set[KnwVertex], Y: Set[KnwVertex]) -> Set[KnwVertex]:
    result = set()
    for q in Y:
        allow_q = class_allow(graph, q, Y)
        for delta in allow_q:
            post_q = set(graph.transition_matrix[q][delta].keys())
            if post_q.issubset(X):
                result.add(q)
                break
    return result


def spre(graph: KnwGraph, Y: Set[KnwVertex]) -> Set[KnwVertex]:
    result = set()
    for v in Y:
        allow_v = class_allow(graph, v, Y)
        if len(allow_v) > 0:
            result.add(v)
    return result


def find_winning_set(graph: KnwGraph, target_observations: List[KnwObs]):
    Y = graph.vertices
    
    Bt = set()
    for o in target_observations:
        Bt = Bt.union(graph.get_observable_vertices(o))
    
    while True:
        X = set()
        while True:
            X_ = apre(graph, X, Y).union(Bt)

            if X_ == X:
                break
            X = deepcopy(X_)

        Y_ = X
        if Y_ == Y:
            break
        Y = deepcopy(Y_)

    return Y

## Ranking ##
def clean_graph(graph: KnwGraph, target_obs: Set[KnwObs], winning_set: Set[KnwVertex]) -> KnwGraph:
    new_graph = KnwGraph(graph.initial_state, winning_set, graph.actions, dict())
    
    for (obs, vertices) in graph.observables_to_v.items():
        z = vertices.intersection(winning_set)
        if len(z) > 0:
            new_graph.observables_to_v[obs] = z
            for v in z:
                new_graph.v_to_observable[v] = obs

    # calculate rankings
    visited = set()
    new_graph.rankings[0] = set()
    for obs in target_obs:
        for v in graph.observables_to_v[obs]:
            assert v in winning_set
            new_graph.rankings[0].add(v)
            new_graph.v_to_rankings[v] = 0
            visited.add(v)

    current_ranking = 1
    while visited != winning_set:
        for v in winning_set:
            if v not in visited:
                deltas = class_allow(graph, v, winning_set)
                if len(deltas) > 0:
                    for delta in deltas:
                        if set(graph.transition_matrix[v][delta].keys()).issubset(new_graph.rankings[current_ranking-1]):
                            if current_ranking not in new_graph.rankings.keys():
                                new_graph.rankings[current_ranking] = deepcopy(new_graph.rankings[current_ranking-1])
                            new_graph.rankings[current_ranking].add(v)
                            new_graph.v_to_rankings[v] = current_ranking
                            visited.add(v)
        current_ranking +=1
   
    for (obs, vertices) in new_graph.observables_to_v.items():
        deltas = class_allow(graph, list(vertices)[0], winning_set)
        for delta in deltas:
            post_vertices = set()
            for vertex in vertices:
                post_vertices = post_vertices.union(set(graph.transition_matrix[vertex][delta]).keys())

            min_rank = min([new_graph.v_to_rankings[v] for v in post_vertices])
            pre_max_rank = max([new_graph.v_to_rankings[v] for v in vertices])
            if min_rank < pre_max_rank:

                for vertex in vertices:
                    if len(graph.transition_matrix[vertex][delta].keys()) == 1 and (vertex in graph.transition_matrix[vertex][delta].keys()):
                        pass
                    else:
                        if vertex not in new_graph.transition_matrix.keys():
                            new_graph.transition_matrix[vertex] = dict()
                        assert delta not in new_graph.transition_matrix[vertex].keys()
                        new_graph.transition_matrix[vertex][delta] = deepcopy(graph.transition_matrix[vertex][delta])

    new_winning_set = set()
    for v in new_graph.transition_matrix.keys():
        new_winning_set.add(v)

    for obs in target_obs:
        for v in graph.observables_to_v[obs]:
            assert isinstance(v, int)
            new_winning_set.add(v)

    winning_set = new_winning_set
    new_graph.vertices = winning_set
    for obs in new_graph.observables_to_v.keys():
        new_graph.observables_to_v[obs] = new_graph.observables_to_v[obs].intersection(winning_set)
    return new_graph
