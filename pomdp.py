from cmath import isclose
from ibm_noise_models import Instruction, MeasChannel, NoiseModel, QuantumChannel
from qmemory import get_seq_probability, handle_write
from qpu_utils import Op
from utils import Precision, Queue
from qstates import QuantumState
from cmemory import ClassicalState, cwrite
from typing import Any, Tuple, List, Dict

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

class POMDPAction:
    def __init__(self, name: str, instruction_sequence: List[Instruction]) -> None:
        self.name = name
        self.instruction_sequence = instruction_sequence

    def __handle_measure_instruction(self, instruction: Instruction, channel: MeasChannel, vertex: POMDPVertex, is_meas1: bool =True, result: Dict[POMDPVertex, float]=None) :
        """applies a measurement instruction to a given hybrid state (POMDP vertex)

        Args:
            instruction (Instruction): an measurement instruction
            channel (MeasChannel): measurement channel for the instruction
            vertex (POMDPVertex): current vertex for which we want to know the successors when we apply the instruction and the channel
            is_meas1 (bool, optional): _description_. is this a measurement 1 or 0?
            result (Dict[POMDPVertex, float], optional): _description_. This is a dictionary that maps a pomdp vertex and the probability of reaching it from the current vertex. We accumulate the result in this dictionary, this is why it is a parameter.

        """        
        assert isinstance(channel, MeasChannel)
        gatedata = instruction.get_gate_data(is_meas_0=(not is_meas1))
        q, meas_prob = get_seq_probability(vertex.quantum_state, [gatedata])

        if meas_prob > 0.0:
            classical_state0 = cwrite(vertex.classical_state, Op.WRITE0, instruction.target)
            classical_state1 = cwrite(vertex.classical_state, Op.WRITE1, instruction.target)

            if is_meas1:
                new_vertex_correct = POMDPVertex(q, classical_state1) # we receive the correct outcome
                new_vertex_incorrect = POMDPVertex(q, classical_state0)
            else:
                new_vertex_correct = POMDPVertex(q, classical_state0) # we receive the correct outcome
                new_vertex_incorrect = POMDPVertex(q, classical_state1)

            if new_vertex_correct not in result.keys():
                result[new_vertex_correct] = 0.0
            if new_vertex_incorrect not in result.keys():
                result[new_vertex_incorrect] = 0.0

            result[new_vertex_correct] += meas_prob * channel.get_ind_probability(is_meas1, is_meas1)
            result[new_vertex_incorrect] += meas_prob * channel.get_ind_probability( is_meas1, not is_meas1)
            assert isclose(channel.get_ind_probability(is_meas1, is_meas1) + channel.get_ind_probability(is_meas1, not is_meas1), 1, rel_tol=Precision.rel_tol )

    def __handle_unitary_instruction(self, instruction: Instruction, channel: QuantumChannel, vertex: POMDPVertex, result: Dict[POMDPVertex, float]=None):
        """_summary_

        Args:
            instruction (Instruction): _description_
            channel (QuantumChannel): _description_
            vertex (POMDPVertex): _description_
            result (Dict[POMDPVertex, float], optional): _description_. Defaults to None.
        """
        
        for (index, err_seq) in enumerate(channel.errors): 
            new_qs = handle_write(vertex.quantum_state, instruction.get_gate_data())
            errored_seq, seq_prob = get_seq_probability(new_qs, err_seq)
            if seq_prob > 0.0:
                new_vertex = POMDPVertex(errored_seq, vertex.classical_state)
                if new_vertex not in result.keys():
                    result[new_vertex] = 0.0
                result[new_vertex] += seq_prob * channel.probabilities[index]

    def __dfs(self, noise_model: NoiseModel, current_vertex: POMDPVertex, index_ins: int) -> Dict[POMDPVertex, float]:
        """perform a dfs to compute successors states of the sequence of instructions.
        It applies the instruction at index self.instructions_seq[index_ins] along with errors recursively

        Args:
            noise_model (NoiseModel): hardware noise model
            current_vertex (POMDPVertex): 
            index_ins (int): should be less than or equal (base case that returns empty dictionary)

        Returns:
            Dict[POMDPVertex, float]: returns a dictionary where the key is a successors POMDPVertex and the corresponding probability of reaching it from current_vertex
        """     
        assert isinstance(current_vertex, POMDPVertex)  
        if index_ins == len(self.instruction_sequence):
            return {current_vertex: 1.0}
        assert index_ins < len(self.instruction_sequence)

        current_instruction = self.instruction_sequence[index_ins]
        instruction_channel = noise_model.instructions_to_channel[current_instruction]

        temp_result = dict()
        if current_instruction.is_meas_instruction():
            # get successors for 0-measurements
            self.__handle_measure_instruction(current_instruction, instruction_channel, current_vertex, is_meas1=False, result=temp_result)

            # get successors for 1-measurements
            self.__handle_measure_instruction(current_instruction, instruction_channel, current_vertex, is_meas1=True, result=temp_result)
        else:
            self.__handle_unitary_instruction(current_instruction, instruction_channel, current_vertex, result=temp_result)

        result = dict()
        for (successor, prob) in temp_result.items():
            successors2 = self.__dfs(noise_model, successor, index_ins=index_ins+1)
            for (succ2, prob2) in successors2.items():
                if succ2 not in result.keys():
                    result[succ2] = 0.0
                result[succ2] += prob*prob2
        assert len(temp_result.keys()) > 0
        if not isclose(sum(result.values()), 1.0, rel_tol=Precision.rel_tol):
            raise Exception(f"Probabilities sum={sum(result.values())} ({self.instruction_sequence[index_ins]}): {result}")
        return result

    def get_successor_states(self, noise_model: NoiseModel, current_vertex: POMDPVertex) -> Dict[POMDPVertex, float]:
        return self.__dfs(noise_model, current_vertex, 0)


class POMDP:
    states: List[POMDPVertex]
    actions: List[POMDPAction]
    transition_matrix: Dict[POMDPVertex, Dict[str, Dict[POMDPVertex, float]]]
    observations: List[ClassicalState]
    initial_state: POMDPVertex

    def __init__(self, initial_state: POMDPVertex, states: List[POMDPVertex], actions: List[POMDPAction], 
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
    
    def serialize(self, problem_instance: Any, output_path: str):
        f = open(output_path, "w")
        f.write("BEGINPOMDP\n")
        f.write(f"INITIALSTATE: {self.initial_state.id}\n")
        vertices_line = ",".join([str(s.id) for s in self.states])
        f.write(f"STATES: {vertices_line}\n")

        # computing target vertices
        target_vertices = []
        for v in self.states:
            if problem_instance.is_target_qs((v.quantum_state, v.classical_state)):
                target_vertices.append(v.id)
        target_v_line = ",".join([str(v) for v in target_vertices])
        f.write(f"TARGETV: {target_v_line}\n")

        # gamma: vertex -> observable
        gamma_line = ",".join([str(s.id)+":" + str(s.classical_state) for s in self.states])
        f.write(f"GAMMA: {gamma_line}\n")
        f.write("BEGINACTIONS\n")
        for action in self.actions:
            f.write(f"{action.name}\n")
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

def default_guard(vertex: POMDPVertex, embedding: Dict[int, int], action: POMDPAction):
    assert isinstance(vertex, POMDPVertex)
    assert isinstance(embedding, dict)
    assert isinstance(action, POMDPAction)
    return True

def build_pomdp(actions: List[POMDPAction],
                noise_model: NoiseModel, 
                horizon: int,
                embedding: Dict[int, int],
                initial_state: Tuple[QuantumState, ClassicalState] = None,
                initial_distribution: List[
                    Tuple[Tuple[QuantumState, ClassicalState], float]]=None, guard: Any = default_guard) -> POMDP:
    """_summary_

    Args:
        channels (QuantumChannel): A list of quantum channels
        initial_state (Tuple[QuantumState, ClassicalState]): A hybrid state
        horizon (int): max horizon to explore
        initial_distribution (List[Tuple[Tuple[QuantumState, ClassicalState], float]], optional): A list of pairs in which the first element of the pair is a hybrid state, while the second element of the pair is a probability denoting the probability of transition from initial_state to the initial distribution. Defaults to None.
        guard (Any): POMDPVertex X embedding X action -> {true, false} says whether in the current set of physical qubits (embedding) of the current vertex (POMDPVertex), an action is permissible.
    """
    if initial_state is None:
        initial_state = (QuantumState(0, qubits_used=list(embedding.values())), ClassicalState())
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
        assert current_v not in graph.keys()
        if current_v not in graph.keys():
            graph[current_v] = dict()

        for action in actions:
            assert isinstance(action, POMDPAction)
            if guard(current_v, embedding, action):
                assert action.name not in graph[current_v].keys()
                graph[current_v][action.name] = dict()

                successors = action.get_successor_states(noise_model, current_v)
                assert len(successors) > 0
                for (succ, prob) in successors.items():
                    assert isinstance(succ, POMDPVertex)
                    
                    new_vertex = create_new_vertex(all_vertices, succ.quantum_state, succ.classical_state)
                    # assert new_vertex not in graph[current_v][action.name].keys()
                    if new_vertex not in graph[current_v][action.name].keys():
                        graph[current_v][action.name][new_vertex] = 0.0
                    graph[current_v][action.name][new_vertex] += prob
                    if new_vertex not in visited:
                        if current_horizon + 1 < horizon:
                            q.push((new_vertex, current_horizon + 1))

    result = POMDP(initial_v, all_vertices, actions, graph)
    return result

