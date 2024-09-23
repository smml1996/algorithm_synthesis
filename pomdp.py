from cmath import isclose
from ibm_noise_models import Instruction, MeasChannel, NoiseModel, QuantumChannel
from qmemory import get_seq_probability, handle_write
from qpu_utils import GateData, NoisyInstruction, Op
from utils import Precision, Queue, Guard, get_kraus_matrix_probability
from qstates import QuantumState
from cmemory import ClassicalState, cwrite
from typing import Tuple, List, Dict

INIT_CHANNEL = "INIT_"
class POMDPVertex:
    local_counter = 1
    def __init__(self, quantum_state: QuantumState, classical_state: ClassicalState):
        assert isinstance(quantum_state, QuantumState)
        assert isinstance(classical_state, ClassicalState)
        self.id = POMDPVertex.local_counter
        POMDPVertex.local_counter += 1
        self.quantum_state = quantum_state
        self.classical_state = classical_state

    def __hash__(self):
        return self.id
    
    def __eq__(self, other):
        return (self.quantum_state == other.quantum_state) and (self.classical_state == other.classical_state)
    
    def __str__(self) -> str:
        return f"V(id={self.id}, {self.quantum_state}, {self.classical_state})"

    def __repr__(self):
        return str(self)
    
class POMDP:
    states: List[POMDPVertex]
    actions: List[QuantumChannel]
    transition_matrix: Dict[POMDPVertex, Dict[str, Dict[POMDPVertex, float]]]
    observations: List[ClassicalState]
    initial_state: POMDPVertex

    def __init__(self, initial_state: POMDPVertex, states: List[POMDPVertex], actions: List[QuantumChannel], 
                transition_matrix: Dict[POMDPVertex, Dict[str, Dict[POMDPVertex, float]]]) -> None:
        assert(isinstance(initial_state, POMDPVertex))
        self.initial_state = initial_state
        self.states = states
        self.actions = actions
        self.transition_matrix = transition_matrix

    def print_graph(self):
        for (v_source, v_source_dict) in self.transition_matrix.items():
            for (channel, v_target_dict) in v_source_dict.items():
                for (v_target, prob) in v_target_dict.items():
                    print(v_source, channel, v_target, prob)

    def get_obs(self, state: POMDPVertex) -> ClassicalState:
        return state.classical_state
    
    def serialize(self, is_target_qs, output_path):
        f = open(output_path, "w")
        f.write("BEGINPOMDP\n")
        f.write(f"INITIALSTATE: {self.initial_state.id}\n")
        vertices_line = ",".join([str(s.id) for s in self.states])
        f.write(f"STATES: {vertices_line}\n")

        # computing target vertices
        target_vertices = []
        for v in self.states:
            if is_target_qs((v.quantum_state, v.classical_state)):
                target_vertices.append(v.id)
        target_v_line = ",".join([str(v) for v in target_vertices])
        f.write(f"TARGETV: {target_v_line}\n")

        # gamma: vertex -> observable
        gamma_line = ",".join([str(s.id)+":" + str(s.classical_state) for s in self.states])
        f.write(f"GAMMA: {gamma_line}\n")
        f.write("BEGINACTIONS\n")
        for channel in self.actions:
            instructions = ",".join([f'Instruction.{i.name}' for i in channel.instructions])
            controls = ",".join([str(c) for c in channel.get_controls()])
            if controls == "":
                controls = "-"
            targets = ",".join([str(t) for t in channel.get_addresses()])
            if targets == "":
                targets = "-"
            f.write(f"{channel.name} {instructions} {controls} {targets}\n")
        f.write("ENDACTIONS\n")
        
        for (fromv, fromv_dict) in self.transition_matrix.items():
            for (channel, channel_dict) in fromv_dict.items():
                for (tov, prob) in channel_dict.items():
                    f.write(f"{fromv.id} {channel} {tov.id} {float(prob):7f}\n")

        f.write("ENDPOMDP\n")
        f.close()

def create_new_vertex(all_vertices, quantum_state, classical_state):
    for v in all_vertices:
        if (v.quantum_state == quantum_state) and (v.classical_state == classical_state):
            return v
    v = POMDPVertex(quantum_state, classical_state)
    all_vertices.append(v)
    return v

def is_vertex_restricted(hybrid_state: POMDPVertex, guards: List[Guard], channel: QuantumChannel, address_space):
    for guard in guards:
        if guard.guard(hybrid_state.quantum_state, hybrid_state.classical_state, address_space):
            if channel in guard.target_channels:
                return True
    return False


def build_pomdp(instructions: List[Instruction],
                noise_model: NoiseModel, 
                initial_state: Tuple[QuantumState, ClassicalState], 
                address_space,
                horizon: int,
                initial_distribution: List[
                    Tuple[Tuple[QuantumState, ClassicalState], float]]=None, guards: List[Guard] = []) -> POMDP:
    """_summary_

    Args:
        channels (QuantumChannel): A list of quantum channels
        initial_state (Tuple[QuantumState, ClassicalState]): A hybrid state
        horizon (int): max horizon to explore
        initial_distribution (List[Tuple[Tuple[QuantumState, ClassicalState], float]], optional): A list of pairs in which the first element of the pair is a hybrid state, while the second element of the pair is a probability denoting the probability of transition from initial_state to the initial distribution. Defaults to None.
    """    

    if not isclose(sum([x for (_, x) in initial_distribution]), 1.0, rel_tol=Precision.rel_tol):
        raise Exception("Initial distribution must sum to 1")
    
    # graph is a dictionary that maps an origin vertex, a channel (str), to another target vertex  and a float which is the probability of transition from the origin vertex to the target vertex
    graph: Dict[POMDPVertex, Dict[str, Dict[POMDPVertex, float]]] = dict()
    all_vertices: List[POMDPVertex] = []

    q  = Queue()

    initial_v = create_new_vertex(all_vertices, initial_state[0], initial_state[1])
    if initial_distribution is None:
        q.push((initial_v, 0)) # second element denotes that this vertex is at horizon 0
    else:
        graph[initial_v] = dict()
        graph[initial_v][INIT_CHANNEL] = dict()
        for (hybrid_state, prob) in initial_distribution:
            v =  create_new_vertex(all_vertices, hybrid_state[0], hybrid_state[1])
            assert v not in graph[initial_v][INIT_CHANNEL].keys()
            graph[initial_v][INIT_CHANNEL][v] = prob
            q.push((v, 0)) # second element denotes that this vertex is at horizon 0

    visited = []
    while not q.is_empty():
        current_v, current_horizon = q.pop()
        if (current_horizon == horizon) or (current_v in visited):
            continue
        visited.append(current_v)
        if current_v not in graph.keys():
            graph[current_v] = dict()

        for noisy_instruction in instructions:
            assert isinstance(noisy_instruction, Instruction)
            if not is_vertex_restricted(current_v, guards, noisy_instruction, address_space):
                # if noisy_instruction.name not in graph[current_v].keys():
                assert noisy_instruction.op.name not in graph[current_v]
                graph[current_v][noisy_instruction.op.name] = dict()

                channel = noise_model.get_instruction_channel(noisy_instruction)
                
                # push to queue (new vertex, horizon)
                if noisy_instruction.is_meas_instruction():
                    assert isinstance(channel, MeasChannel)

                    gatedata_p0 = noisy_instruction.get_gate_data(is_meas_0=True)
                    gatedata_p1 = noisy_instruction.get_gate_data(is_meas_1=True)

                    q0, meas0_prob = get_seq_probability(current_v.quantum_state, [gatedata_p0])
                    q1, meas1_prob = get_seq_probability(current_v.quantum_state, [gatedata_p1])

                    classical_state0 = cwrite(current_v.classical_state, Op.WRITE0, noisy_instruction.target)
                    classical_state1 = cwrite(current_v.classical_state, Op.WRITE1, noisy_instruction.target)

                    if meas0_prob > 0.0:
                        new_vertex = create_new_vertex(all_vertices, q0, classical_state0)
                        assert new_vertex not in graph[current_v][channel.name].keys()
                        graph[current_v][channel.name][new_vertex] = meas0_prob * channel.get_ind_probability(0, 0)
                        if new_vertex not in visited:
                            if current_horizon + 1 < horizon:
                                q.push((new_vertex, current_horizon + 1))

                        new_vertex = create_new_vertex(all_vertices, q0, classical_state1)
                        assert new_vertex not in graph[current_v][channel.name].keys()
                        graph[current_v][channel.name][new_vertex] = meas0_prob * channel.get_ind_probability(0, 1)
                        if new_vertex not in visited:
                            if current_horizon + 1 < horizon:
                                q.push((new_vertex, current_horizon + 1))

                    if meas1_prob > 0.0:
                        new_vertex = create_new_vertex(all_vertices, q1, classical_state1)
                        assert new_vertex not in graph[current_v][channel.name].keys()
                        graph[current_v][channel.name][new_vertex] = meas1_prob * channel.get_ind_probability(1, 1)
                        if new_vertex not in visited:
                            if current_horizon + 1 < horizon:
                                q.push((new_vertex, current_horizon + 1))

                        new_vertex = create_new_vertex(all_vertices, q1, classical_state0)
                        assert new_vertex not in graph[current_v][channel.name].keys()
                        graph[current_v][channel.name][new_vertex] = meas1_prob * channel.get_ind_probability(1, 0)
                        if new_vertex not in visited:
                            if current_horizon + 1 < horizon:
                                q.push((new_vertex, current_horizon + 1))
                else:
                    assert isinstance(channel, QuantumChannel)
                    new_qs = handle_write(current_v.quantum_state, noisy_instruction.get_gate_data())
                    for (index, err_seq) in enumerate(channel.errors): 
                        errored_seq, seq_prob = get_seq_probability(new_qs, err_seq)
                        if seq_prob > 0.0:
                            new_vertex = create_new_vertex(all_vertices, errored_seq, current_v.classical_state)
                            assert new_vertex not in graph[current_v][channel.name].keys()
                            graph[current_v][channel.name][new_vertex] = seq_prob * channel.probabilities[index]
                            if new_vertex not in visited:
                                if current_horizon + 1 < horizon:
                                    q.push((new_vertex, current_horizon + 1))

    result = POMDP(initial_v, all_vertices, instructions, graph)
    return result

