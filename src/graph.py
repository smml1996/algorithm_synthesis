from copy import deepcopy
from math import floor, log2, isclose
from typing import Any, Dict, List, Optional, Set
import importlib

import os, sys
sys.path.append(os.getcwd()+"/..")
from cmemory import ClassicalState, cwrite
from qmemory import handle_write, get_seq_probability
from qstates import QuantumState, is_quantum_state_entangled
from utils import Belief, Vertex, QuantumChannel, is_classical_op, GateData, Precision, Queue, Instruction, meas_instructions


class LowLevelGraph:
    vertices: Set
    observables_to_v: Dict[int, Set]  # maps observable to set of vertices that belong to that observation
    adj_list: Dict[int, Dict[str, Dict]]  # vertex_index --> channel --> outgoing edges
    channels_data: Dict[str, Any]
    initial_obs: Optional[int]
    rankings: Dict[int, Set]
    v_to_rankings: Dict[int, int]
    adj_probs: Dict[int, Dict[str, Belief]]
    obs_adj_list: Dict[int, Dict[str, Dict[int, int]]]  # obs -> Channel -> seq_index -> obs

    def __init__(self) -> None:
        self.vertices = set()
        self.observables_to_v = dict()
        self.adj_list = dict()
        self.channels_data = dict()
        self.initial_obs = None
        self.rankings = dict()
        self.v_to_rankings = dict()
        self.adj_probs = dict()

    def get_num_qubits(self):
        result = 0
        for channel in self.channels_data.values():
            for op in channel.gates:
                for gate in op:
                    result = max(result, gate.address + 1)
                    controls = gate.controls
                    if controls is not None:
                        result = max(result, max(controls) + 1)
        return result

    def get_post_vertices(self, observable, channel) -> Set[int]:
        answer = set()
        for vertex in self.observables_to_v[observable]:
            answer = answer.union(self.adj_list[vertex][channel])
        return answer

    def get_deltas_nqubits(self):
        num_deltas = 0
        for adj_d in self.adj_list.values():
            num_deltas = max(num_deltas, len(adj_d.keys()))
        result = floor(log2(num_deltas)) + 1
        return result


class Graph:
    vertices: List[Vertex]
    obs_adj_list: Dict[int, Dict[str, Dict[int, int]]]  # obs -> Channel -> seq_index -> obs
    obs_to_belief: Dict[int, Belief]
    error_obs: List[int]

    def __init__(self,
                 channels: List[QuantumChannel] = None,
                 initial_states: List[QuantumState] = None,
                 debug=False, initial_belief=None, target_belief_threshold=0.99, path=None) -> None:

        if path is not None:
            self.current_obs_count = -1
            self.from_file(path, debug)
        else:
            self.current_obs_count = 1
            self.observables_to_v = dict()  #
            self.channels = channels
            self.observables_to_v[0] = []
            self.vertices = []
            self.error_obs = []
            self.target_obs = []
            self.obs_to_belief = dict()
            self.obs_to_depth = dict()
            for state in initial_states:
                if isinstance(state, Vertex):
                    v = state
                    v.observable = 0
                    v.in_edges = dict()
                    v.out_edges = dict()
                else:
                    qs, cs = state
                    # qs.truncate()
                    assert isinstance(qs, QuantumState)
                    # qs.normalize()
                    v = Vertex(qs, cs)
                self.observables_to_v[0].append(v)
                self.vertices.append(v)
            self.obs_to_depth[0] = 0
            if not (initial_belief is None):
                self.obs_to_belief[0] = initial_belief
            else:
                initial_belief = Belief()
                value = 1.0 / float(len(self.observables_to_v[0]))
                for v in self.observables_to_v[0]:
                    initial_belief.set_val(v.id, value, do_floor=True)
                self.obs_to_belief[0] = initial_belief
            self.obs_adj_list = dict()
            self.adj_probs = dict()
            self.debug = debug
            self.target_belief_threshold = target_belief_threshold

    def dump_graph(self, name, dir_=""):
        file = open(dir_ + name, "w")

        file.write("import os, sys\n")
        file.write('sys.path.append(os.getcwd()+"/../..")\n')
        file.write('sys.path.append(os.getcwd()+"/..")\n')
        file.write("from utils import *\n")
        file.write("\n\n")

        # TODO: dump channels
        channels_d = dict()
        for channel in self.channels:
            assert isinstance(channel, QuantumChannel)
            assert channel.name not in channels_d.keys()
            channels_d[channel.name] = channel.dump_channel()
        file.write("channels = " + channels_d.__str__())
        file.write("\n\n")

        # dump vertices
        temp_v = dict()
        for v in self.vertices:
            assert v.id not in temp_v.keys()
            temp_v[v.id] = v.get_serializable()
        file.write("vertices = " + temp_v.__str__())
        file.write("\n\n")

        # dump observables_to_v 
        temp_observables_to_v = dict()
        for (key, vertices) in self.observables_to_v.items():
            tempv = []
            for v in vertices:
                tempv.append(v.id)
            assert key not in temp_observables_to_v.keys()
            temp_observables_to_v[key] = tempv
        file.write("observables_to_v = " + temp_observables_to_v.__str__())
        file.write("\n\n")

        file.write("error_obs = " + self.error_obs.__str__())
        file.write("\n\n")
        file.write("target_obs = " + self.target_obs.__str__())
        file.write("\n\n")
        file.write("obs_to_belief = " + self.obs_to_belief.__str__())
        file.write("\n\n")

        file.write("obs_adj_list = " + self.obs_adj_list.__str__())
        file.write("\n\n")
        file.write("adj_probs = " + self.adj_probs.__str__())
        file.write("\n\n")
        file.write("target_belief_thresh = " + str(self.target_belief_threshold))
        file.write("\n")
        file.close()

    def from_file(self, path, debug):
        mod = importlib.import_module(path)
        self.vertices = []
        temp_vertices_d = dict()
        for (v_id, v) in mod.vertices.items():
            qs = QuantumState()
            qs.sparse_vector = v['qs']
            cs = ClassicalState()
            cs.sparse_vector = v['cs']
            new_v = Vertex(qs, cs, from_serializable=v)
            self.vertices.append(new_v)
            temp_vertices_d[v['id']] = new_v

        self.observables_to_v = dict()
        for (obs, vertices) in mod.observables_to_v.items():
            assert obs not in self.observables_to_v.keys()
            self.observables_to_v[obs] = []
            for v_id in vertices:
                self.observables_to_v[obs].append(temp_vertices_d[v_id])

        self.channels = []
        for (_, c_data) in mod.channels.items():
            self.channels.append(QuantumChannel(from_serializable=c_data))

        self.error_obs = mod.error_obs
        self.target_obs = mod.target_obs

        self.obs_to_belief = dict()
        for (obs, belief_d) in mod.obs_to_belief.items():
            self.obs_to_belief[obs] = Belief(d=belief_d)
        self.obs_adj_list = mod.obs_adj_list

        self.adj_probs = dict()
        for (index, d2) in mod.adj_probs.items():
            self.adj_probs[index] = dict()
            for (index2, d3) in d2.items():
                self.adj_probs[index][index2] = dict()
                for (channel_name, belief) in d3.items():
                    self.adj_probs[index][index2][channel_name] = Belief(d=belief)
        self.debug = debug
        self.target_belief_threshold = mod.target_belief_thresh

    def obs_exists(self, belief: Belief):
        assert isinstance(belief, Belief)
        for (obs, b) in self.obs_to_belief.items():
            if b == belief:
                return obs
        return None

    def get_vertex_state(self, id_):
        for v in self.vertices:
            if v.id == id_:
                return v.quantum_state

    def get_vertex_by_state(self, qs, cs):
        for v in self.vertices:
            if v.quantum_state == qs and v.classical_state == cs:
                return v
        return None

    @staticmethod
    def get_vertex(vertices, qs, cs) -> Optional[Vertex]:
        for v in vertices:
            assert type(v) == Vertex
            if v.quantum_state == qs and v.classical_state == cs:
                return v
        return None

    def parse_information_sets(self, info_set: List[Vertex], belief, channel, v_to_prev, prev_obs, create_new=False):
        current_obs = self.obs_exists(belief)
        if (current_obs is None) or create_new:
            current_obs = self.current_obs_count
            self.current_obs_count += 1
            self.observables_to_v[current_obs] = info_set
            self.obs_to_belief[current_obs] = belief

        channel_name = channel.name
        for vertex in info_set:
            for current_vertex in v_to_prev[vertex.id]:
                # add in edges to vertex in info set
                if prev_obs not in vertex.in_edges.keys():
                    vertex.in_edges[prev_obs] = dict()
                if not (channel_name in vertex.in_edges[prev_obs].keys()):
                    vertex.in_edges[prev_obs][channel_name] = set()
                vertex.in_edges[prev_obs][channel_name].add((current_obs, current_vertex.id))

                if prev_obs not in current_vertex.out_edges.keys():
                    current_vertex.out_edges[prev_obs] = dict()
                if not (channel_name in current_vertex.out_edges[prev_obs].keys()):
                    current_vertex.out_edges[prev_obs][channel_name] = set()
                # if we are in prev_obs, we use channel_name then we get to (current_obs, vertex.id)
                current_vertex.out_edges[prev_obs][channel_name].add((current_obs, vertex.id))
        return current_obs

    @staticmethod
    def get_channel_gate_transformation(gate_sequence, current_quantum_state, current_cs):
        answer = current_quantum_state
        answer_cs = current_cs
        for op in gate_sequence:
            assert isinstance(op, GateData)
            if is_classical_op(op.label):
                answer_cs = cwrite(answer_cs, op.label, op.address)
            else:
                answer = handle_write(answer, op, is_inverse=False)
            if answer is None:
                return None, None
        return answer, answer_cs

    def process_new_vertex(self, qs: QuantumState, cs: ClassicalState):
        vertex = self.get_vertex_by_state(qs, cs)
        if vertex is None:
            vertex = Vertex(qs, cs)
            self.vertices.append(vertex)
        return vertex

    def get_belief_success_prob(self, belief, is_target_qs, address_space):
        b = deepcopy(belief)
        b.normalize()
        vals = 0.0
        for (v, val) in b.d.items():
            if is_target_qs(self.get_vertex_state(v), address_space):
                vals += val

        return round(vals, Precision.PRECISION)

    def is_target_belief(self, belief, is_target_qs, address_space):
        # succ_prob = self.get_belief_success_prob(belief, is_target_qs, address_space)
        # rel_tol = 1 / (10 ** (Precision.PRECISION + 1))
        return False
    
    def build_graph_from_algorithm(self, algorithm_node):
        q = Queue()
        q.push((0, algorithm_node))
        assert len(self.observables_to_v.keys()) == 1
        # assert obs_limit == 10000 # we are migrating to the use of horizon
        while (not q.is_empty()):
            current_obs, alg = q.pop()
            assert isinstance(current_obs, int)
            current_layer = self.observables_to_v[current_obs]
            current_belief = self.obs_to_belief[current_obs]
            assert current_obs not in self.obs_adj_list.keys()
            assert current_obs not in self.adj_probs.keys()
            self.obs_adj_list[current_obs] = dict()
            self.adj_probs[current_obs] = dict()
            for channel in self.channels:
                if (alg is not None) and (alg.instruction is not None) and alg.instruction in channel.instructions:
                    assert channel.name not in self.obs_adj_list[current_obs].keys()
                    self.obs_adj_list[current_obs][channel.name] = dict()
                    v_to_prev = dict()
                    next_obs_vertices_ = dict()
                    next_beliefs = dict()
                    cs_val_to_seq = dict()
                    
                    for (seq_index, seq) in enumerate(channel.gates):
                        observable_part = seq_index
                        for current_vertex in current_layer:
                            if current_vertex.id not in self.adj_probs[current_obs].keys():
                                self.adj_probs[current_obs][current_vertex.id] = dict()

                            if channel.name not in self.adj_probs[current_obs][current_vertex.id].keys():
                                self.adj_probs[current_obs][current_vertex.id][channel.name] = Belief()
                            assert isinstance(current_vertex, Vertex)
                            new_correctqs, new_correctcs = self.get_channel_gate_transformation(
                                seq,
                                current_vertex.quantum_state,
                                current_vertex.classical_state)
                            if not (new_correctqs is None):
                                # new_correctqs.truncate()
                                cs_val = new_correctcs.get_memory_val()

                                if cs_val not in cs_val_to_seq.keys():
                                    cs_val_to_seq[cs_val] = observable_part
                                    observable_part = (observable_part + 1) % 2

                                if cs_val not in next_obs_vertices_.keys():
                                    assert cs_val not in next_beliefs.keys()
                                    next_obs_vertices_[cs_val] = []
                                    next_beliefs[cs_val] = Belief()

                                # get existing vertex if already exists, otherwise create a new one
                                new_vertex = self.process_new_vertex(new_correctqs, new_correctcs)

                                if new_vertex.id not in v_to_prev.keys():
                                    v_to_prev[new_vertex.id] = []

                                if current_vertex not in v_to_prev[new_vertex.id]:
                                    v_to_prev[new_vertex.id].append(current_vertex)

                                seq_prob = get_seq_probability(current_vertex.quantum_state, seq)
                                next_beliefs[cs_val].add_val(new_vertex.id,
                                                            current_belief.get(current_vertex.id) * seq_prob *
                                                            channel.probabilities[seq_index][0], do_floor=True)

                                self.adj_probs[current_obs][current_vertex.id][channel.name].add_val(new_vertex.id,
                                    seq_prob * channel.probabilities[seq_index][0], do_floor=True)

                                if next_beliefs[cs_val].get(new_vertex.id) > 0:
                                    if new_vertex not in next_obs_vertices_[cs_val]:
                                        next_obs_vertices_[cs_val].append(new_vertex)
                            else:
                                continue

                            for (index, gate_error) in enumerate(channel.errors[seq_index]):
                                new_error_qs, new_error_cs = self.get_channel_gate_transformation(
                                                                    gate_error,
                                                                    current_vertex.quantum_state,
                                                                    current_vertex.classical_state)
                                if not (new_error_qs is None):
                                    # new_error_qs.truncate()
                                    cs_val = new_error_cs.get_memory_val()

                                    new_vertex = self.process_new_vertex(new_error_qs, new_error_cs)

                                    if new_vertex.id not in v_to_prev.keys():
                                        v_to_prev[new_vertex.id] = []

                                    if current_vertex not in v_to_prev[new_vertex.id]:
                                        v_to_prev[new_vertex.id].append(current_vertex)

                                    seq_prob = get_seq_probability(current_vertex.quantum_state, gate_error)
                                    if cs_val not in cs_val_to_seq.keys():
                                        cs_val_to_seq[cs_val] = observable_part
                                        observable_part = (observable_part + 1) % 2
                                    if cs_val not in next_obs_vertices_.keys():
                                        assert cs_val not in next_beliefs.keys()
                                        next_obs_vertices_[cs_val] = []
                                        next_beliefs[cs_val] = Belief()

                                        
                                    next_beliefs[cs_val].add_val(new_vertex.id,
                                                                current_belief.get(current_vertex.id) * seq_prob *
                                                                channel.probabilities[seq_index][index + 1], do_floor=True)

                                    self.adj_probs[current_obs][current_vertex.id][channel.name].add_val(
                                                                                new_vertex.id,
                                                                                seq_prob * channel.probabilities[seq_index]
                                                                                [index + 1], do_floor= True)

                                    if next_beliefs[cs_val].get(new_vertex.id) > 0:
                                        if new_vertex not in next_obs_vertices_[cs_val]:
                                            next_obs_vertices_[cs_val].append(new_vertex)
                            self.adj_probs[current_obs][current_vertex.id][channel.name].normalize(do_floor=True)
                    for (cs_val, next_obs_vertices) in next_obs_vertices_.items():
                        if len(next_obs_vertices) > 0:
                            next_belief = next_beliefs[cs_val]
                            seq_index = cs_val_to_seq[cs_val]
                            assert seq_index == 0  or seq_index == 1
                            assert isinstance(next_belief, Belief)
                            next_obs = self.parse_information_sets(next_obs_vertices, next_belief, channel, v_to_prev, current_obs, create_new=True)
                            if next_obs not in self.obs_to_depth.keys():
                                next_obs_depth = self.obs_to_depth[current_obs]+1
                                self.obs_to_depth[next_obs] = next_obs_depth
                            assert isinstance(next_obs, int)
                            if cs_val not in self.obs_adj_list[current_obs][channel.name].keys():
                                self.obs_adj_list[current_obs][channel.name][seq_index] = next_obs
                            else:
                                assert  self.obs_adj_list[current_obs][channel.name][seq_index] == next_obs
                            if seq_index == 0:
                                if alg.instruction in meas_instructions:
                                    next_alg = alg.case0
                                else:
                                    next_alg = alg.next_ins
                            else:
                                assert seq_index == 1
                                next_alg = alg.case1
                            q.push((next_obs, next_alg))
        return True, len(self.observables_to_v)

    def build_graph(self, is_target_qs, address_space, obs_limit=1000, max_layer_width=10, belief_threshold=0.01, debug=False, horizon=0, do_safety_check=False, is_pomdp=False):
        def m1(address):
            if address == 0:
                return address_space[0]
            assert address == 1
            return address_space[2]
        def m2(address):
            if address == 0:
                return address_space[1]
            assert address == 1
            return address_space[2]
        q = Queue()
        visited_layers = set()
        visited_layers.add(0)  # add observable 0
        q.push(0)
        assert len(self.observables_to_v.keys()) == 1
        # assert obs_limit == 10000 # we are migrating to the use of horizon
        while (not q.is_empty()):
            if len(self.observables_to_v.keys()) > obs_limit:
                return False, len(self.observables_to_v.keys())
            current_obs = q.pop()
            current_norm = sum(self.obs_to_belief[current_obs].d.values())
            if debug:
                print(self.obs_to_depth[current_obs], "~", q.len()," ~ ",len(self.observables_to_v.keys()))
            if self.obs_to_depth[current_obs] == horizon:
                continue
            assert self.obs_to_depth[current_obs] < horizon
            assert isinstance(current_obs, int)
            current_layer = self.observables_to_v[current_obs]
            current_belief = self.obs_to_belief[current_obs]
            assert current_obs not in self.obs_adj_list.keys()
            assert current_obs not in self.adj_probs.keys()
            self.obs_adj_list[current_obs] = dict()
            self.adj_probs[current_obs] = dict()
            for channel in self.channels:
                assert channel.name not in self.obs_adj_list[current_obs].keys()
                self.obs_adj_list[current_obs][channel.name] = dict()
                v_to_prev = dict()
                next_obs_vertices_ = dict()
                next_beliefs = dict()
                cs_val_to_seq = dict()
                
                for (seq_index, seq) in enumerate(channel.gates):
                    observable_part = seq_index
                    for current_vertex in current_layer:
                        ## add entanglement check here
                        if do_safety_check and (Instruction.MEAS in channel.instructions):
                            if is_quantum_state_entangled(current_vertex.quantum_state, m1) or is_quantum_state_entangled(current_vertex.quantum_state, m2):
                                continue
                        if current_vertex.id not in self.adj_probs[current_obs].keys():
                            self.adj_probs[current_obs][current_vertex.id] = dict()

                        if channel.name not in self.adj_probs[current_obs][current_vertex.id].keys():
                            self.adj_probs[current_obs][current_vertex.id][channel.name] = Belief()
                        assert isinstance(current_vertex, Vertex)
                        new_correctqs, new_correctcs = self.get_channel_gate_transformation(
                            seq,
                            current_vertex.quantum_state,
                            current_vertex.classical_state)
                        if not (new_correctqs is None):
                            # new_correctqs.truncate()
                            cs_val = new_correctcs.get_memory_val()

                            if cs_val not in cs_val_to_seq.keys():
                                cs_val_to_seq[cs_val] = observable_part
                                observable_part = (observable_part + 1) % 2

                            if cs_val not in next_obs_vertices_.keys():
                                assert cs_val not in next_beliefs.keys()
                                next_obs_vertices_[cs_val] = []
                                next_beliefs[cs_val] = Belief()

                            # get existing vertex if already exists, otherwise create a new one
                            new_vertex = self.process_new_vertex(new_correctqs, new_correctcs)

                            if new_vertex.id not in v_to_prev.keys():
                                v_to_prev[new_vertex.id] = []

                            if current_vertex not in v_to_prev[new_vertex.id]:
                                v_to_prev[new_vertex.id].append(current_vertex)

                            seq_prob = get_seq_probability(current_vertex.quantum_state, seq)
                            next_beliefs[cs_val].add_val(new_vertex.id,
                                                         current_belief.get(current_vertex.id) * seq_prob *
                                                         channel.probabilities[seq_index][0], do_floor=True)

                            self.adj_probs[current_obs][current_vertex.id][channel.name].add_val(new_vertex.id,
                                seq_prob * channel.probabilities[seq_index][0], do_floor=True)

                            if next_beliefs[cs_val].get(new_vertex.id) > 0:
                                if new_vertex not in next_obs_vertices_[cs_val]:
                                    next_obs_vertices_[cs_val].append(new_vertex)
                        else:
                            continue

                        for (index, gate_error) in enumerate(channel.errors[seq_index]):
                            new_error_qs, new_error_cs = self.get_channel_gate_transformation(
                                                                gate_error,
                                                                current_vertex.quantum_state,
                                                                current_vertex.classical_state)
                            if not (new_error_qs is None):
                                # new_error_qs.truncate()
                                cs_val = new_error_cs.get_memory_val()

                                new_vertex = self.process_new_vertex(new_error_qs, new_error_cs)

                                if new_vertex.id not in v_to_prev.keys():
                                    v_to_prev[new_vertex.id] = []

                                if current_vertex not in v_to_prev[new_vertex.id]:
                                    v_to_prev[new_vertex.id].append(current_vertex)

                                seq_prob = get_seq_probability(current_vertex.quantum_state, gate_error)
                                if cs_val not in cs_val_to_seq.keys():
                                    cs_val_to_seq[cs_val] = observable_part
                                    observable_part = (observable_part + 1) % 2
                                if cs_val not in next_obs_vertices_.keys():
                                    assert cs_val not in next_beliefs.keys()
                                    next_obs_vertices_[cs_val] = []
                                    next_beliefs[cs_val] = Belief()

                                    
                                next_beliefs[cs_val].add_val(new_vertex.id,
                                                             current_belief.get(current_vertex.id) * seq_prob *
                                                             channel.probabilities[seq_index][index + 1], do_floor=True)

                                self.adj_probs[current_obs][current_vertex.id][channel.name].add_val(
                                                                            new_vertex.id,
                                                                            seq_prob * channel.probabilities[seq_index]
                                                                            [index + 1], do_floor= True)

                                if next_beliefs[cs_val].get(new_vertex.id) > 0:
                                    if new_vertex not in next_obs_vertices_[cs_val]:
                                        next_obs_vertices_[cs_val].append(new_vertex)
                        self.adj_probs[current_obs][current_vertex.id][channel.name].normalize(do_floor=True)
                for (cs_val, next_obs_vertices) in next_obs_vertices_.items():
                    if len(next_obs_vertices) > 0:
                        next_belief = next_beliefs[cs_val]
                        seq_index = cs_val_to_seq[cs_val]
                        assert isinstance(next_belief, Belief)
                        if is_pomdp:
                            next_belief.normalize()
                        next_obs = self.parse_information_sets(next_obs_vertices, next_belief, channel, v_to_prev, current_obs)
                        if next_obs not in self.obs_to_depth.keys():
                            next_obs_depth = self.obs_to_depth[current_obs]+1
                            self.obs_to_depth[next_obs] = next_obs_depth
                        assert isinstance(next_obs, int)
                        if cs_val not in self.obs_adj_list[current_obs][channel.name].keys():
                            self.obs_adj_list[current_obs][channel.name][seq_index] = next_obs
                        else:
                            assert  self.obs_adj_list[current_obs][channel.name][seq_index] == next_obs
                        if not (next_obs in visited_layers):
                            visited_layers.add(next_obs)
                            is_obs_valid = self.is_obs_valid(next_obs, is_target_qs, address_space)
                            belief_val = next_belief.get_sum()
                            if is_obs_valid or self.is_target_belief(next_belief, is_target_qs, address_space):
                                self.target_obs.append(next_obs)
                            elif belief_val <= belief_threshold:
                                self.error_obs.append(next_obs)
                            elif len(next_obs_vertices) < max_layer_width:
                                q.push(next_obs)
                                if len(self.observables_to_v.keys()) > obs_limit:
                                    return False, len(self.observables_to_v)
        return True, len(self.observables_to_v)

    def print_obs_vertices(self, is_target_qs, m):
        for (obs, vertices) in self.observables_to_v.items():
            print(f"***** Observable: {obs} ********")
            for v in vertices:
                print(f"{float(self.obs_to_belief[obs].get(v.id))} ~~ {v.quantum_state} ~~ {v.classical_state} ~~ {is_target_qs(v.quantum_state, m)}")

    def print_obs_succ_rates(self, is_target_qs, address_space, obs_list=-1):
        if obs_list == -1:
            for (obs, belief) in self.obs_to_belief.items():
                succ_rate = self.get_belief_success_prob(belief, is_target_qs, address_space)
                print(f"{obs} --> {succ_rate}")
        else:
            assert isinstance(obs_list, list)
            for obs in obs_list:
                belief = self.obs_to_belief[obs]
                succ_rate = self.get_belief_success_prob(belief, is_target_qs, address_space)
                print(f"{obs} --> {succ_rate}")

    def get_low_level_graph(self) -> LowLevelGraph:
        new_graph = LowLevelGraph()
        new_graph.adj_probs = self.adj_probs
        new_graph.obs_adj_list = self.obs_adj_list
        new_graph.initial_obs = 0
        new_graph.vertices = set([v.id for v in self.vertices])
        for channel in self.channels:
            new_graph.channels_data[channel.name] = channel

        for (key, vertices) in self.observables_to_v.items():
            assert not (key in new_graph.observables_to_v.keys())
            new_graph.observables_to_v[key] = set([v.id for v in vertices])

        for v in self.vertices:
            assert not (v.id in new_graph.adj_list.keys())
            new_graph.adj_list[v.id] = dict()
            for (obs, channel_dict) in v.out_edges.items():
                assert obs not in new_graph.adj_list[v.id].keys()
                new_graph.adj_list[v.id][obs] = dict()
                for (channel, vertices) in channel_dict.items():
                    new_graph.adj_list[v.id][obs][channel] = set([v2 for v2 in vertices])
        return new_graph

    def is_obs_valid(self, obs, is_target_qs, address_mapping):
        for v in self.observables_to_v[obs]:
            if not is_target_qs(v.quantum_state, address_mapping):
                return False
        return True

    def get_target_obs(self, is_target_qs, address_mapping):
        answer = []
        for o in self.observables_to_v.keys():
            if self.is_obs_valid(o, is_target_qs, address_mapping):
                answer.append(o)
        return answer

    def get_target_vertices(self, is_target_qs, address_mapping):
        answer = set()
        for v in self.vertices:
            if is_target_qs(v.quantum_state, address_mapping):
                assert v.id not in answer
                answer.add(v.id)
        return answer
