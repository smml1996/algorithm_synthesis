from cmath import isclose
from ibm_noise_models import Instruction, MeasChannel, NoiseModel, QuantumChannel
from qmemory import get_seq_probability, handle_write
from qpu_utils import Op
from utils import Precision, Queue
from qstates import QuantumState
from cmemory import ClassicalState, cwrite
from typing import Any, Optional, Set, Tuple, List, Dict

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
        self.symbols = []
        
        # define order of symbols as they appear in the instruction sequence
        used_symbols = set()
        for instruction in instruction_sequence:
            if instruction.symbols is not None:
                for symbol in instruction.symbols:
                    if symbol not in used_symbols:
                        self.symbols.append(symbol)
                        used_symbols.add(symbol)
                        
    def optimize(self):
        new_instruction_sequence = []
        for instruction in self.instruction_sequence:
            if not self.instruction.is_identity():
                new_instruction_sequence.append(instruction)
                    
    def bind_symbols_from_lst(self, lst: List[float]) -> Any:
        """_summary_

        Args:
            lst (List[float]): _description_

        Returns:
            Any: returns a POMDP action with symbols binded
        """ 
        assert len(lst) == len(self.symbols)
        d = zip(self.symbols, lst)
        return self.bind_symbols_from_dict(d)
        
    def bind_symbols_from_dict(self, d: Dict[str, float]) -> Any:
        """_summary_

        Args:
            d (Dict[str, float]): _description_

        Returns:
            Any: returns a POMDP action with symbols binded
        """        
        
        new_instruction_seq = []
        
        for instruction in self.instruction_sequence:
            new_instruction_seq.append(instruction.bind_symbols_from_dict(d))
        return POMDPAction(self.name, new_instruction_seq)
            
    def get_num_parameters(self):
        return len(self.symbols)

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
            
            prob = meas_prob * channel.get_ind_probability(is_meas1, is_meas1)
            if prob > 0:
                if new_vertex_correct not in result.keys():
                    result[new_vertex_correct] = 0.0
                result[new_vertex_correct] += prob

            prob = meas_prob * channel.get_ind_probability( is_meas1, not is_meas1)
            if prob > 0:
                if new_vertex_incorrect not in result.keys():
                    result[new_vertex_incorrect] = 0.0
                result[new_vertex_incorrect] += prob
            assert isclose(channel.get_ind_probability(is_meas1, is_meas1) + channel.get_ind_probability(is_meas1, not is_meas1), 1, rel_tol=Precision.rel_tol)

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
                
    def __handle_reset_instruction(self, instruction: Instruction, channel: QuantumChannel, vertex: POMDPVertex, is_meas1: bool =True, result: Dict[POMDPVertex, float]=None):
        """_summary_

        Args:
            instruction (Instruction): _description_
            channel (QuantumChannel): _description_
            vertex (POMDPVertex): _description_
            result (Dict[POMDPVertex, float], optional): _description_. Defaults to None.
        """
        assert instruction.op == Op.RESET
        meas_instruction = Instruction(instruction.target, Op.MEAS)
        for (index, err_seq) in enumerate(channel.errors): 
            new_qs, prob_new_qs = get_seq_probability(vertex.quantum_state, [meas_instruction.get_gate_data(is_meas_0=(not is_meas1))])
            if prob_new_qs > 0:
                if is_meas1:
                    x_instruction = Instruction(instruction.target, Op.X)
                    new_qs = handle_write(new_qs, x_instruction.get_gate_data())
                errored_seq, seq_prob = get_seq_probability(new_qs, err_seq)
                seq_prob = prob_new_qs * seq_prob
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
        temp_result = dict()
        if current_instruction.is_classical():
            new_classical_state = cwrite(current_vertex.classical_state, current_instruction.op, current_instruction.target)
            new_vertex = POMDPVertex(current_vertex.quantum_state, new_classical_state)
            temp_result[new_vertex] = 1.0
        else:
            instruction_channel = noise_model.get_instruction_channel(current_instruction)
            if current_instruction.is_meas_instruction():
                # get successors for 0-measurements
                self.__handle_measure_instruction(current_instruction, instruction_channel, current_vertex, is_meas1=False, result=temp_result)

                # get successors for 1-measurements
                self.__handle_measure_instruction(current_instruction, instruction_channel, current_vertex, is_meas1=True, result=temp_result)
            elif current_instruction.op == Op.RESET:
                # WARNING: use of reset not known in all models, check when using real hardware specifications
                self.__handle_reset_instruction(current_instruction, instruction_channel, current_vertex, is_meas1=False, result=temp_result)
                
                self.__handle_reset_instruction(current_instruction, instruction_channel, current_vertex, is_meas1=True, result=temp_result)
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
    
    def get_target(self):
        return self.instruction_sequence[-1].target

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

    def get_observable_vertices(self, observable) -> Set[POMDPVertex]:
        if observable in self.observables_to_v.keys():
            return self.observables_to_v[observable]
        answer = set()
        for vertex in self.states:
            if vertex.classical_state == observable:
                answer.add(vertex)
        self.observables_to_v[observable] = answer
        return self.observables_to_v[observable]
        
    def get_low_level_adj_matrix(self):
        ''' used for imperfect information games
        '''
        answer = dict()
        for (src_vertex, succ_dict) in self.transition_matrix.items():
            assert src_vertex.id not in answer.keys()
            answer[src_vertex.id] = dict()
            for (action_name, succ2_dict) in succ_dict.items():
                assert action_name not in answer[src_vertex.id].keys()
                for (succ, _) in succ2_dict.items():
                    answer[src_vertex.id][action_name] = succ.id
        return answer
    
    def get_v_id_to_classical_state(self):
        ''' used for imperfect information games
        '''
        answer = dict()
        for v in self.states:
            answer[v.id] = v.classical_state
        return answer
    
    def serialize(self, problem_instance: Any, output_path: str):
        f = open(output_path, "w")
        f.write("BEGINPOMDP\n")
        f.write(f"INITIALSTATE: {self.initial_state.id}\n")
        vertices_line = ",".join([str(s.id) for s in self.states])
        f.write(f"STATES: {vertices_line}\n")

        # computing target vertices
        target_vertices = []
        for v in self.states:
            r = problem_instance.get_reward((v.quantum_state, v.classical_state))
            target_vertices.append(f"{v.id}:{r}")
        target_v_line = ",".join(target_vertices)
        f.write(f"REWARDS: {target_v_line}\n")

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
        
    def get_reversed_digraph(self):
        digraph = dict()
        for (fromv, fromv_dict) in self.transition_matrix.items():
            for (channel, channel_dict) in fromv_dict.items():
                for (tov, prob_) in channel_dict.items():
                    if tov not in digraph.keys():
                        digraph[tov] = dict()  
                    if channel not in digraph[tov].keys():
                        digraph[tov][channel] = []
                    digraph[tov][channel].append(fromv)
        return digraph
        
    def optimize_graph(self, problem_instance) -> bool:
        
        new_states = set() # this also contains the set of visited states
        
        # identify target states
        q = Queue()
        for v in self.states:
            if problem_instance.get_reward((v.quantum_state, v.classical_state)) != 0:
                q.push(v)
                new_states.add(v)
        
        rev_digraph = self.get_reversed_digraph()
        while not q.is_empty():
            current_v = q.pop()
            assert current_v in new_states
            if current_v in rev_digraph.keys():
                for (channel_, from_vertices) in rev_digraph[current_v].items():
                    for from_v in from_vertices:
                        if from_v not in new_states:
                            # we push all states that have non-zero probability of reaching the target set, or a state that have non-zero probability of reaching the target set
                            q.push(from_v)
                            new_states.add(from_v)
        
        if self.initial_state not in new_states:
            self.transition_matrix = dict()
            self.states = []
            # self.actions = []
            return False
        
        
        new_transition_matrix = dict()
        
        for (fromv, fromv_dict) in self.transition_matrix.items():
            if fromv in new_states:
                # determine if channel leads to some vertex we are interested
                assert fromv not in new_transition_matrix.keys()
                for (channel, channel_dict) in fromv_dict.items():
                    for (tov, prob) in channel_dict.items():
                        if tov in new_states:
                            if fromv not in new_transition_matrix.keys():
                                new_transition_matrix[fromv] = dict()

                            if channel not in new_transition_matrix[fromv].keys():
                                new_transition_matrix[fromv][channel] = dict()
                                
                            new_transition_matrix[fromv][channel][tov] = prob
                    
        self.states = new_states
        self.transition_matrix = new_transition_matrix
        return True
            
            
            
def get_vertex(all_vertices, quantum_state, classical_state) -> Optional[POMDPVertex]:
    for v in all_vertices:
        if (v.quantum_state == quantum_state) and (v.classical_state == classical_state):
            return v
    return None

def create_new_vertex(all_vertices, quantum_state, classical_state):
    v = get_vertex(all_vertices, quantum_state, classical_state)
    if v is None:
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
                    Tuple[Tuple[QuantumState, ClassicalState], float]]=None, guard: Any = default_guard,
                qubits_used=None) -> POMDP:
    """_summary_

    Args:
        channels (QuantumChannel): A list of quantum channels
        initial_state (Tuple[QuantumState, ClassicalState]): A hybrid state
        horizon (int): max horizon to explore
        initial_distribution (List[Tuple[Tuple[QuantumState, ClassicalState], float]], optional): A list of pairs in which the first element of the pair is a hybrid state, while the second element of the pair is a probability denoting the probability of transition from initial_state to the initial distribution. Defaults to None.
        guard (Any): POMDPVertex X embedding X action -> {true, false} says whether in the current set of physical qubits (embedding) of the current vertex (POMDPVertex), an action is permissible.
    """
    if initial_state is None:
        if qubits_used is None:
            initial_state = (QuantumState(0, qubits_used=list(embedding.values())), ClassicalState())
        else:
            initial_state = (QuantumState(0, qubits_used=qubits_used), ClassicalState())
    
    # graph is a dictionary that maps an origin vertex, a channel (str), to another target vertex  and a float which is the probability of transition from the origin vertex to the target vertex
    graph: Dict[POMDPVertex, Dict[str, Dict[POMDPVertex, float]]] = dict()
    all_vertices: List[POMDPVertex] = []

    q  = Queue()

    initial_v = create_new_vertex(all_vertices, initial_state[0], initial_state[1])
    if initial_distribution is None:
        q.push((initial_v, 0)) # second element denotes that this vertex is at horizon 0
    else:
        if not isclose(sum([x for (_, x) in initial_distribution]), 1.0, rel_tol=Precision.rel_tol):
            raise Exception("Initial distribution must sum to 1")
        graph[initial_v] = dict()
        graph[initial_v][INIT_CHANNEL] = dict()
        for (hybrid_state, prob) in initial_distribution:
            v =  create_new_vertex(all_vertices, hybrid_state[0], hybrid_state[1])
            assert v not in graph[initial_v][INIT_CHANNEL].keys()
            graph[initial_v][INIT_CHANNEL][v] = prob
            q.push((v, 0)) # second element denotes that this vertex is at horizon 0

    visited = set()
    while not q.is_empty():
        
        current_v, current_horizon = q.pop()
        # print(q.len(), current_horizon)
        if horizon != -1:
            if (current_horizon == horizon):
                continue
        if current_v in visited:
            continue
        visited.add(current_v)
        # assert current_v not in graph.keys()
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
                        if (horizon == -1) or (current_horizon + 1 < horizon):
                            q.push((new_vertex, current_horizon + 1))

    result = POMDP(initial_v, all_vertices, actions, graph)
    return result

