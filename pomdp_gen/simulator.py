
from cmath import isclose
from typing import List, Optional
from cmemory import ClassicalState, cread, cwrite
from qmemory import QuantumState, get_qs_probability, handle_write, is_multiqubit_gate
from utils import AlgorithmNode, Gate, GateData, algorithm_exists, choose_from_prob_distribution, Instruction, default_mapping, instruction_to_gate, Precision
import random
from noise import NoiseData, NoiseModel
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np

class QSimulator:
    qmemory: QuantumState
    meas_cache: ClassicalState
    noise_model: NoiseModel
    count_executed_instructions: int
    log: List[str]
    instructions_applied: List
    def __init__(self, noise_model = NoiseModel(None, None), seed=None, with_readout_err=True) -> None:
        self.log = []
        self.instructions_applied = []
        if not (seed is None):
            self.log.append(f"SEED: {seed}")
            random.seed(seed)
        self.noise_model = noise_model
        self.qmemory = QuantumState(0)
        self.meas_cache = ClassicalState()
        self.count_executed_instructions = 0
        self.with_readout_err = with_readout_err
        

    def apply_noise_model(self, instruction: Instruction, target: int, controls: List):
        noise_found = False
        for noise in self.noise_model.noise:
            assert isinstance(noise, NoiseData)
            if (instruction in noise.target_instructions):
                    if is_multiqubit_gate(instruction_to_gate(instruction)):
                        assert len(controls) == 1
                        assert noise.controls is not None
                        if (noise.qubits == -1 or (target in noise.qubits)) and (noise.controls==-1 or (set(controls).issubset(noise.controls))):
                            if noise_found:
                                print("WARNING: Applying more than one noise model")
                            noise_found = True
                            e_index = choose_from_prob_distribution([i for i in range(len(noise.apply_instructions))], noise.probabilities)
                            e = noise.apply_instructions[e_index]
                            self.log.append(f"NOISE: {e} ({instruction} {target}) ")
                            for (index, e_) in enumerate(e):
                                if noise.qubits != -1:
                                    t = noise.params[e_index][index]
                                else:
                                    if index % 2 == 0:
                                        t = controls[0]
                                    else:
                                        t = target
                                self.qmemory = handle_write(self.qmemory, GateData(instruction_to_gate(e_), t, None))

                    else:
                        if noise.qubits == -1 or (target in noise.qubits):
                            if noise_found:
                                print("WARNING: Applying more than one noise model")
                            noise_found = True
                            e_index = choose_from_prob_distribution([i for i in range(len(noise.apply_instructions))], noise.probabilities)
                            e = noise.apply_instructions[e_index]
                            self.log.append(f"NOISE: {e} ({instruction} {target}) ")
                            for (index, e_) in enumerate(e):
                                if noise.qubits != -1:
                                    target_ = noise.params[e_index][index]
                                    assert target == target_
                                self.qmemory = handle_write(self.qmemory, GateData(instruction_to_gate(e_), target, None))

        # apply readout error:
        if self.with_readout_err and (self.noise_model.readout_error is not None):
            if instruction in [Instruction.MEAS, Instruction.MEASX]:
                for readout_err in self.noise_model.readout_error:
                    if (readout_err.qubits == -1) or (target in readout_err.qubits):
                        if noise_found:
                            print("WARNING: Applying more than one noise model")
                        noise_found = True
                        assert len(readout_err.probabilities) == 2
                        assert len(readout_err.apply_instructions) == 2
                        assert Instruction.X in readout_err.apply_instructions
                        assert Instruction.I in readout_err.apply_instructions
                        prev_value = bool(cread(self.meas_cache, target))
                        assert prev_value == 0 or prev_value == 1
                        probabilities = readout_err.probabilities[prev_value]
                        e = choose_from_prob_distribution(readout_err.apply_instructions, probabilities)
                        if e == Instruction.X:
                            self.log.append(f"NOISE: FLIP MEAS {target} ")
                            self.meas_cache.insert(target, not prev_value)
        # if not noise_found:
        #     print(f"WARNING: seems no noise model was found for instruction {instruction} {target} {controls}")

    def are_instructions_equal(self, index:int, instruction: Instruction, target: int, controls: List[int] = None):
        if self.instructions_applied[index][0] != instruction.name:
            return False
        
        if self.instructions_applied[index][1] != target:
            return False
        
        if self.instructions_applied[index][2] != controls:
            return False
        return True

    def apply_instruction(self, instruction: Instruction, target: int, controls: List[int] = None, params=None, with_noise=True) -> Optional[int]:
        self.instructions_applied.append([instruction.name, target, controls])
        return_val = None
        if (instruction != Instruction.MEAS) and (instruction != Instruction.MEASX):
            self.log.append(f"QPU: {instruction} {target} {controls}")
            self.qmemory = handle_write(self.qmemory, GateData(instruction_to_gate(instruction), target, controls, params=params))
        else:
            # if controls is not None:
            #     raise Exception("MEAS instruction should not have control qubits")
            
            if instruction == Instruction.MEASX:
                self.qmemory = handle_write(self.qmemory, GateData(Gate.H, target, controls, params=params))

            prob0 = get_qs_probability(self.qmemory, target, is_zero=True, is_floor=False)
            prob1 = 1.0 - prob0
            assert isclose(prob0 + prob1, 1.0, rel_tol=Precision.rel_tol)
            projector = choose_from_prob_distribution([Gate.P0, Gate.P1], [prob0, prob1])
            self.log.append(f"QPU: {instruction.name} {projector} {target}")

            
            self.qmemory = handle_write(self.qmemory, GateData(projector, target, []))
            if projector == Gate.P0:
                self.meas_cache = cwrite(self.meas_cache, Gate.WRITE0, target)
            else:
                assert projector == Gate.P1
                self.meas_cache = cwrite(self.meas_cache, Gate.WRITE1, target)

            if instruction == Instruction.MEASX:
                self.qmemory = handle_write(self.qmemory, GateData(Gate.H, target, controls, params=params))
        
        self.count_executed_instructions += 1
        if with_noise:
            self.apply_noise_model(instruction, target, controls)
        if instruction in [Instruction.MEAS, Instruction.MEASX]:
            return_val = cread(self.meas_cache, target)
        return return_val
    
    def apply_instructions(self, instructions: List[Instruction], targets: int, controls: List[int] = None, params=None) -> Optional[int]:
        outcome = None
        for (i, instruction) in enumerate(instructions):
            target = targets[i]
            control = controls[i]
            temp_outcome = self.apply_instruction(instruction, target, [control], params=params)
            if temp_outcome != None:
                outcome = temp_outcome
        return outcome
    
    def h(self, target):
        self.apply_instruction(Instruction.H, target, None)

    def cx(self, control, target):
        assert isinstance(control, int)
        assert isinstance(control, int)
        self.apply_instruction(Instruction.CNOT, target, [control])


    def get_meas_cache_val(self, address) -> bool:
        return cread(self.meas_cache, address)

def execute_traditional_algorithm(noise_model, algorithm, is_target_qs, address_space, shots=100):
    accuracy = 0
    exec_ins = 0
        
    for i in range(shots):
        result = algorithm(noise_model, i, address_space)
        if is_target_qs(result.qmemory, address_space):
            accuracy +=1
        exec_ins += result.count_executed_instructions
    return accuracy/float(shots), exec_ins/float(shots)


def execute_randomized_algorithm(noise_model, procedures, channel_error_probs, l_adj_events, is_target_qs, weights, belief_threshold=1, shots=100, l_address_space=-1, is_debug=False, initial_qs=None, return_qpu=False, seed=1, print_algorithms=False):
    """_summary_

    Args:
        noise_model (NoiseModel): _description_
        procedures (_type_): algorithms autogenerated to call
        channel_error_probs (_type_): _description_
        l_adj_events (_type_): _description_
        is_target_qs (bool): function that returns True when a quantum state is a target quantum state.
        belief_threshold (int, optional): _description_. Defaults to 1.
        shots (int, optional): _description_. Defaults to 100.
        l_address_space (int, optional): _description_. Defaults to -1.
        is_debug (bool, optional): _description_. Defaults to False.
        initial_qs (_type_, optional): _description_. Defaults to None.

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """

    assert len(procedures) == len(l_adj_events)
    accuracy = 0
    exec_ins = 0
    qpu = None
    diff_algorithms = []
    for shot in range(shots):
        if initial_qs is not None:
            qpu = QSimulator(noise_model, (shot+1)*seed)
            qpu.qmemory = initial_qs
        else:
            qpu = None
        init_belief = None
        for (index, procedure) in enumerate(procedures):
            if l_address_space == -1:
                address_space = default_mapping()
            else:
                address_space = l_address_space[index]
            result = procedure(noise_model, weights, seed=(shot+1)*seed, belief_threshold=belief_threshold, init_belief=init_belief, address_space=address_space, qpu=qpu)
            qpu = QSimulator(noise_model, (shot+1)*seed)
            qpu.qmemory = result.qmemory
            qpu.meas_cache = result.meas_cache
            qpu.log = result.log
            qpu.instructions_applied = result.instructions_applied
            qpu.count_executed_instructions += result.n_instructions
            exec_ins += result.n_instructions
            init_belief = result.belief
            if result.done:
                break
        if print_algorithms:
            if not algorithm_exists(diff_algorithms, qpu.instructions_applied):
                diff_algorithms.append(qpu.instructions_applied)
        if return_qpu:
            assert accuracy == 0
            return qpu
        if is_target_qs(qpu.qmemory, address_space):
            accuracy +=1
        elif is_debug:
            raise Exception("Target quantum state not reached.")
        # else:
        #     print("******")
        #     print(qpu.qmemory)
        #     print(qpu.log)
    for algo in diff_algorithms:
        print(algo)
        print("--------")
    return accuracy/float(shots), exec_ins/float(shots)

def superposition_assertion(qpu: QSimulator, qubit: int, ancillae: int) -> int:
    # https://dl.acm.org/doi/pdf/10.1145/3373376.3378488
    val = qpu.get_meas_cache_val(ancillae)
    
    if val == 1:
        # we flip the ancillae if its current state is 1
        qpu.apply_instruction(Instruction.X, ancillae, None)
    else:
        assert val == 0
    qpu.apply_instruction(Instruction.H, ancillae, None)
    qpu.apply_instruction(Instruction.CNOT, qubit, [ancillae], None)
    qpu.apply_instruction(Instruction.H, ancillae, None)
    return qpu.apply_instruction(Instruction.MEAS, ancillae, None)

def robust_read(qpu: QSimulator, instructions: List[Instruction], target, controls):
    outcome1 = qpu.apply_instructions(instructions, target, controls)

    outcome2 = qpu.apply_instructions(instructions, target, controls)

    while outcome1 != outcome2:
        outcome1 = outcome2
        outcome2 = qpu.apply_instructions(instructions, target, controls)

    return outcome1

def execute_algorithm_nodes(noise_model, seed, head: AlgorithmNode):
    qpu = QSimulator(noise_model, seed)

    while True:
        is_meas = (Instruction.MEAS in head.instructions) or (Instruction.MEASX in head.instructions)
        outcome = qpu.apply_instructions(head.instructions)
        if is_meas:
            assert head.next_ins is None
            if outcome == 0:
                head = head.case0
            else:
                head = head.case1
        else:
            assert head.case0 is None
            assert head.case1 is None
            head = head.next_ins
        if head is None:
            break
    return qpu.qmemory, qpu.count_executed_instructions

def get_mapped_list(m_name, elements):
    is_first = True
    answer = "["
    for e in elements:
        if is_first:
            is_first = False
        else:
            answer += ", "
        answer += f"{m_name}({e})"
    answer += "]"
    return answer

def dump_algorithm_node(node: AlgorithmNode, tabs="\t"):
    if node is None:
        return f"{tabs}pass\n"
    curr_ins = node.instructions.__str__()
    targets = get_mapped_list("m", node.targets)
    controls = get_mapped_list("m", node.controls)
    params = node.params.__str__()
    answer = f"{tabs}outcome = qpu.apply_instructions({curr_ins}, {targets}, {controls}, {params})\n"
    if (Instruction.MEASX in node.instructions) or (Instruction.MEAS in node.instructions):
        case0 = dump_algorithm_node(node.case0, tabs+"\t")
        case1 = dump_algorithm_node(node.case1, tabs+"\t")
        answer += f"{tabs}if outcome == 0:\n"
        answer += case0
        answer += f"{tabs}else:\n"
        answer += f"{tabs}\tassert outcome == 1\n"
        answer += case1
    else:
        answer += dump_algorithm_node(node.next_ins, tabs)

    return answer

def dump_algorithm_nodes(alg_node, alg_name):
    answer = f"def {alg_name}(qpu, m):\n"
    answer += dump_algorithm_node(alg_node)

    return answer
        

def is_program_valid_for_state(algorithm, initial_state, is_target_qs, m, seed, noise_model=NoiseModel(None, None)):
    qpu = QSimulator(noise_model, seed)
    qpu.qmemory = initial_state
    head = algorithm
    while head is not None:
        if head.controls is None:
            cls = []
            for _ in head.instructions:
                cls.append(None)
        else:
            cls = []
            for c in head.controls:
                cls.append(m(c))

        targets = []
        for t in head.targets:
            targets.append(m(t))
        
        outcome = qpu.apply_instructions(head.instructions, targets, cls, head.params)
        if (Instruction.MEAS in head.instructions) or (Instruction.MEASX in head.instructions):
            if outcome == 0:
                head = head.case0
            else:
                head = head.case1
        else:
            head = head.next_ins
    return is_target_qs(qpu.qmemory, m)

def is_program_valid(algorithm, initial_states, is_target_qs, m, shots=1000, thresh=0.75):
    acc = 0
    for initial_state in initial_states:
        for s in range(shots):
            acc += is_program_valid_for_state(algorithm, initial_state, is_target_qs, m, s)
    acc /= (shots*len(initial_states))
    if acc > thresh:
        return True
    return False
    

def ibm_simulate_circuit(qc: QuantumCircuit, noise_model, shots, initial_layout, seed=1):
    # Create noisy simulator backend
    sim_noise = AerSimulator(method ='statevector', noise_model=noise_model)

    # Transpile circuit for noisy basis gates
    circ_tnoise = transpile(qc, sim_noise, optimization_level=0, initial_layout=initial_layout)
    # Run and get counts
    
    result = sim_noise.run(circ_tnoise, run_options={"shots":2000, "seed_simulator": seed}).result()
    
    return np.asarray(result.data()['res'])

