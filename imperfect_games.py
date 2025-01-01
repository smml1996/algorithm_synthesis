from typing import Dict, List, Set, Tuple
from cmemory import ClassicalState
from pomdp import POMDP, POMDPVertex
from copy import deepcopy

## FINDING WINNING SET ##
def allow(graph: POMDP, v: POMDPVertex, Y: Set[POMDPVertex]) -> Set[str]:
    result = set()
    for (channel, vertices_dict) in graph.transition_matrix[v].items():
        vertices = set(vertices_dict.keys())
        if vertices.issubset(Y):
            result.add(channel)
    return result
        

def class_allow(graph: POMDP, v: POMDPVertex, Y: Set[POMDPVertex]) -> Set[int]:
    observable = v.classical_state
    equivalence_class = deepcopy(graph.get_observable_vertices(observable))
    result = allow(graph, v, Y)
    equivalence_class.remove(v)
    for q in equivalence_class:
        result = result.intersection(allow(graph, q, Y))
    return result

def apre(graph: POMDP, X: Set[POMDPVertex], Y: Set[POMDPVertex]) -> Set[POMDPVertex]:
    result = set()
    for q in Y:
        allow_q = class_allow(graph, q, Y)
        for delta in allow_q:
            post_q = set(graph.transition_matrix[q][delta].keys())
            if post_q.issubset(X):
                result.add(q)
                break
    return result


def spre(graph: POMDP, Y: Set[POMDPVertex]) -> Set[POMDPVertex]:
    result = set()
    for v in Y:
        allow_v = class_allow(graph, v, Y)
        if len(allow_v) > 0:
            result.add(v)
    return result


def find_winning_set(graph: POMDP, target_observations: List[ClassicalState]):
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
def clean_graph(graph: POMDP, target_obs: Set[ClassicalState], winning_set: Set[POMDPVertex]) -> POMDP:
    new_graph = POMDP(graph.initial_state, winning_set, graph.actions, dict())
    
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