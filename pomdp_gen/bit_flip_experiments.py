import importlib
from cmath import isclose
from math import ceil
import random

from sympy import conjugate, simplify
from qiskit import IBMQ
import os, sys
from noise import NoiseModel, get_ibm_noise_model
from pomdp import build_pomdp
from utils import  *
from game import get_bellman_graph, get_real_accuracy
from graph import Graph
from simulator import QSimulator, ibm_simulate_circuit
from qstates import QuantumState
from typing import List
from cmemory import ClassicalState
from copy import deepcopy
import time
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from notebooks.notebook_utils import DIR_PREFIX

MAX_HORIZON = 7
BOTTOM_HORIZON = 5
MAX_PRECISION = 30

backends_w_embs = [
    # 5-qubit backends
    FAKE_ATHENS, 
    FAKE_BELEM, FAKE_TENERIFE,
    FAKE_LIMA, FAKE_ROME, FAKE_MANILA, 
    FAKE_SANTIAGO, FAKE_BOGOTA, FAKE_OURENSE, FAKE_YORKTOWN,
    FAKE_ESSEX, FAKE_VIGO, FAKE_BURLINGTON, FAKE_QUITO, FAKE_LONDON,

    # 7-qubit backends
    FAKE_JAKARTA, FAKE_OSLO, FAKE_PERTH, 
    FAKE_LAGOS, FAKE_NAIROBI, FAKE_CASABLANCA,

    # 14-qubit backend
    FAKE_MELBOURNE,

    # 16-qubit backend
    FAKE_GUADALUPE,

    # 20-qubit backend
    FAKE_TOKYO, FAKE_POUGHKEEPSIE, FAKE_JOHANNESBURG, FAKE_BOEBLINGEN,
    FAKE_ALMADEN, FAKE_SINGAPORE,

    # 27-qubit backend
    FAKE_MUMBAI, FAKE_PARIS, FAKE_AUCKLAND, FAKE_KOLKATA, FAKE_TORONTO, FAKE_MONTREAL, FAKE_SYDNEY, FAKE_CAIRO, FAKE_HANOI, FAKE_GENEVA,

    # 28-qubit backend
    FAKE_CAMBRIDGE,

    # 53-qubit backend
    FAKE_ROCHESTER,

    # 65-qubit backend
    FAKE_BROOKLYN, FAKE_MANHATTAN,

    # 127-qubit backend
    FAKE_WASHINGTON# no successs probability
]

def is_control_entangled(quantum_state: QuantumState, address_space):
    simulator = QSimulator()
    simulator.qmemory = deepcopy(quantum_state)
    simulator.apply_instruction(Instruction.MEAS, address_space[2])
    if is_bell_state(simulator.qmemory, address_space):
        return False
    return True

def are_controls_entangled(quantum_state, classical_state, address_space):
    return is_control_entangled(quantum_state, address_space)

class ReadoutNoise:
    def __init__(self, target, success0, success1):
        self.target = target
        self.success0 = success0
        self.success1 = success1
        self.diff = success0 - success1
        self.acc_err = 1-success0 + 1-success1
        self.abs_diff = abs(success0-success1)

    def __str__(self):
        return f"{self.target}, {self.diff}, {self.acc_err}, {self.success0}, {self.success1}"
    def __repr__(self):
        return self.__str__()
    
def get_pivot_qubits(noise_model):
    result = set()
    noises = []
    for qubit in range(noise_model.get_num_qubits()):
        if noise_model.get_qubit_indegree(qubit) > 1:
            noise_data = noise_model.get_qubit_readout_error(qubit)
            probs = noise_data[0].probabilities
            success0 = probs[0][1]
            success1 = probs[1][1]
            noises.append(ReadoutNoise(qubit, success0, success1))
    temp = sorted(noises, key=lambda x : x.success0)
    result.add(temp[0].target)
    result.add(temp[len(temp)-1].target)

    temp = sorted(noises, key=lambda x : x.success1)
    result.add(temp[0].target)
    result.add(temp[len(temp)-1].target)

    temp = sorted(noises, key=lambda x: x.acc_err) # accumulated error
    result.add(temp[0].target)
    result.add(temp[len(temp)-1].target)

    temp = sorted(noises, key=lambda x: x.diff)
    if temp[0].diff != temp[len(temp)-1].diff:
        result.add(temp[0].target)
        result.add(temp[len(temp)-1].target)

    temp = sorted(noises, key=lambda x: x.abs_diff)
    if temp[0].abs_diff != temp[len(temp)-1].abs_diff:
        result.add(temp[0].target)
        assert (temp[0].abs_diff < temp[len(temp)-1].abs_diff)
    return result

def  is_bell_state(qs, m):
    if is_target_qs(qs, m):
        return True
    
    rho = qs.get_density_matrix(m)
    pt = partial_trace(rho,[4,2], [1])
    is_bell_state = True
    for i in range(4):
        for j in range(4):
            if i == j and (i==1 or i == 2):
                if not isclose(pt[i][j], 0.5, abs_tol=Precision.isclose_abstol):
                    is_bell_state = False
            elif (i == 1 and j == 2) or (j == 2 and i == 1):
                if not isclose(pt[i][j],0.5, abs_tol=Precision.isclose_abstol):
                    is_bell_state = False
            elif not isclose(pt[i][j], 0.0, abs_tol=Precision.isclose_abstol):
                is_bell_state = False
    
    if is_bell_state:
        return True
    
    is_bell_state = True
    for i in range(4):
        for j in range(4):
            if i == j and (i==1 or i == 2):
                if not isclose(pt[i][j], 0.5, abs_tol=Precision.isclose_abstol):
                    is_bell_state = False
            elif (i == 1 and j == 2) or (j == 2 and i == 1):
                if not isclose(pt[i][j],-0.5, abs_tol=Precision.isclose_abstol):
                    is_bell_state = False
            elif not isclose(pt[i][j], 0.0, abs_tol=Precision.isclose_abstol):
                is_bell_state = False
    return is_bell_state
    

    
def get_selected_couplers(noise_model, target):
    couplers = noise_model.get_qubit_couplers(target)
    first_pair = (couplers[0], couplers[1])
    second_pair = (couplers[len(couplers) -1], couplers[len(couplers) -2])
    return [first_pair, second_pair]

def does_result_contains_d(result, d):
    for d_ in result:
        controls1 = set([d[0], d[1]])
        controls2 = set([d_[0], d_[1]])
        if d_[2] == d[2] and controls1 == controls2:
            return True
    return False

def get_backend_embeddings(backend: str):
    result = []
    noise_model = get_ibm_noise_model(backend)
    if noise_model.get_num_qubits() < 14:
        pivot_qubits = set()
        for qubit in range(noise_model.get_num_qubits()):
            if noise_model.get_qubit_indegree(qubit) > 1:
                pivot_qubits.add(qubit)
    else:
        pivot_qubits = get_pivot_qubits(noise_model)
    for target in pivot_qubits:
        assert(isinstance(target, int))
        for p in get_selected_couplers(noise_model, target):
            d_temp = dict()
            d_temp[0] = p[0][0]
            d_temp[1] = p[1][0]
            d_temp[2] = target
            if not does_result_contains_d(result, d_temp):
                result.append(deepcopy(d_temp))
    return result

def is_target_qs(qs, address_space):
    """
    returns true if is in either of the following Bell states:
    1/sqrt(2)(|00> + |11>)
    1/sqrt(2)(|00> - |11>)
    Parameters
    ----------
    qs
    address_space

    Returns
    -------

    """

    assert isinstance(qs, QuantumState)

    rho = qs.get_density_matrix(address_space)
    assert rho.shape[0] == rho.shape[1]
    assert rho.shape[0] == 8
    reduced_rho = partial_trace(rho,[4,2], [1])
    assert reduced_rho.shape[0] == reduced_rho.shape[1]
    assert reduced_rho.shape[0] == 4
    is_bell_state = True
    for i in range(4):
        for j in range(4):
            if i == j and (i==0 or i == 3):
                if not isclose(reduced_rho[i][j], 0.5, abs_tol=Precision.isclose_abstol):
                    is_bell_state = False
            elif (i == 0 and j == 3) or (j == 0 and i == 3):
                if not isclose(reduced_rho[i][j], 0.5, abs_tol=Precision.isclose_abstol):
                    is_bell_state = False
            elif not isclose(reduced_rho[i][j], 0.0, abs_tol=Precision.isclose_abstol):
                is_bell_state = False
    
    if is_bell_state:
        return True
    
    is_bell_state = True
    for i in range(4):
        for j in range(4):
            if i == j and (i==0 or i == 3):
                if not isclose(reduced_rho[i][j], 0.5, abs_tol=Precision.isclose_abstol):
                    is_bell_state = False
            elif (i == 0 and j == 3) or (j == 0 and i == 3):
                if not isclose(reduced_rho[i][j],-0.5, abs_tol=Precision.isclose_abstol):
                    is_bell_state = False
            elif not isclose(reduced_rho[i][j], 0.0, abs_tol=Precision.isclose_abstol):
                is_bell_state = False
    
    return is_bell_state
                


def get_bell_state(i, m) -> QuantumState:
    # It prepares one of the 4 bell states
    assert 0 <= i < 4
    qpu = QSimulator(NoiseModel(None, None), 1)
    qpu.apply_instruction(Instruction.H, m[0])
    qpu.apply_instruction(Instruction.CNOT, m[1], [m[0]])
    if i == 0:
        # prepare |00> + |11>
        pass
    elif i == 1:
        # prepare |01> + |10>
        qpu.apply_instruction(Instruction.X, m[0])
    elif i == 2:
        # prepare |00> - |11>
        qpu.apply_instruction(Instruction.Z, m[0])
    else:
        # prepare |01> - |10>
        assert i == 3
        qpu.apply_instruction(Instruction.X, m[0])
        qpu.apply_instruction(Instruction.Z, m[0])

    return qpu.qmemory

def ibm_prepare_bell_state(qc, bell_index):
    qc.h(0)
    qc.cx(0, 1)
    if bell_index == 1:
        qc.x(0)
    elif bell_index == 2:
        qc.z(0)
    elif bell_index == 3:
        qc.x(0)
        qc.z(0)
    else:
        assert(bell_index == 0)

def ibm_is_target_qs(statev,  layout):
    sorted_layout = deepcopy(layout)
    sorted_layout.sort()
    pos0 = get_position(sorted_layout, layout[0])
    assert pos0 is not None
    pos1 = get_position(sorted_layout, layout[1])
    assert pos1 is not None

    prob00 = 0.0
    prob11 = 0.0
    for basis in range(8):
        value = statev[basis]
        b_str = int_to_bin(basis)
        while len(b_str) < 3:
            b_str += "0"
        assert len(b_str) == 3
        qubit0 = (b_str[pos0] == '1')
        qubit1 = (b_str[pos1] == '1')
        if qubit0 != qubit1:
            probv = value * conjugate(value)
            if isinstance(probv, complex):
                probv = probv.real
            if not isclose(probv, 0.0, abs_tol=Precision.isclose_abstol): 
                return False
        else:
            if qubit0 == 0:
                prob00 += value*conjugate(value)
            else:
                prob11 += value*conjugate(value)

    if isinstance(prob00, complex):
        prob00 = simplify(prob00.real)
        
    if isinstance(prob11, complex):
        prob11 = simplify(prob11.real)
        
    if isclose(prob00, 0.0, abs_tol=Precision.isclose_abstol):
        return False
    
    return True
    
    
def ibm_execute_my_algo(shots, alg, backend, mapping, is_simulated=True, log=None):
    shots_per_initial_state = shots/4
    accuracy = 0
    initial_layout = [mapping[0], mapping[1], mapping[2]]
    ibm_noise_model = get_ibm_noise_model(backend, for_ibm=True)
    basis_gates = ibm_noise_model.basis_gates

    assert isinstance(backend, str)
    real_hardware_name = backend.lower().replace("fake", "ibm")
    
    if log is not None:
        log_file = open(log, "w")
        log_file.write(f"shots={shots}\n")
    for bell_state_index in range(4):
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        qc = QuantumCircuit(qr, cr)
        ibm_prepare_bell_state(qc, bell_state_index)
        if isinstance(alg, AlgorithmNode):
            execute_algorithm(alg, qc, True, basis_gates=basis_gates, cbits=cr)
        else:
            alg(qc, basis_gates, cr)
        if is_simulated:
            qc.save_statevector('res', pershot=True)
            state_vs = ibm_simulate_circuit(qc, ibm_noise_model, shots_per_initial_state, initial_layout)
            for state in state_vs:
                if ibm_is_target_qs(state, initial_layout):
                    accuracy += 1
        else:
            IBMQ.load_account()
            # Get the available providers
            providers = IBMQ.providers()
            # Choose a provider
            provider = providers[0]
            assert log is not None
            backend = provider.get_backend(real_hardware_name)
            job = execute(qc, backend, shots=shots_per_initial_state, initial_layout=initial_layout, optimization_level=0)
            log_file.write(f"{job.job}\n")
    if log is not None:
        log_file.close()
    return round(accuracy/(1024*4), 3)

def execute_my_algo(shots, algorithm_node, noise_model, is_target_qs, address_space):
    acc = 0
    ins_sum = 0

    for i in range(shots):
        state = get_bell_state(random.randint(0, 3), address_space)
        qpu = QSimulator(noise_model, seed=i)
        qpu.qmemory = state
        n_instructions = execute_algorithm(algorithm_node, qpu, for_ibm=False, address_space=address_space)
        ins_sum += n_instructions
        if is_target_qs(qpu.qmemory, address_space):
            acc += 1

    acc = round(acc / shots, Precision.PRECISION)
    avg_ins = round(ins_sum / shots, Precision.PRECISION)
    return acc, avg_ins

def get_initial_states(m) -> List[QuantumState]:
    """

    Parameters
    ----------
    m : is an address space mapping for logic qubit to a physical qubit in the actual QPU
    """
    # this functions returns a
    initial_states = []
    initial_cs = ClassicalState()
    qpu = QSimulator(NoiseModel(None, None), 1)
    qpu.apply_instruction(Instruction.H, m[0])
    qpu.apply_instruction(Instruction.CNOT, m[1], [m[0]])
    initial_states.append((qpu.qmemory, initial_cs))

    qpu.apply_instruction(Instruction.X, m[0])
    initial_states.append((qpu.qmemory, initial_cs))
    qpu.apply_instruction(Instruction.X, m[0])

    qpu.apply_instruction(Instruction.Z, m[0])
    initial_states.append((qpu.qmemory, initial_cs))
    qpu.apply_instruction(Instruction.Z, m[0])

    qpu.apply_instruction(Instruction.Z, m[0])
    qpu.apply_instruction(Instruction.X, m[0])
    initial_states.append((qpu.qmemory, initial_cs))
    return initial_states

def get_belief_and_initial_states(m, is_target_prob=0.5):
    initial_states = get_initial_states(m)
    v_initial_states  = []
    initial_belief = Belief()
    for (qs, cs) in initial_states:
        v = Vertex(qs, cs)
        v_initial_states.append(v)
        if is_target_qs(qs, m):
            initial_belief.set_val(v.id, is_target_prob/2)
        else:
            initial_belief.set_val(v.id, (1-is_target_prob)/2)
    return initial_belief, v_initial_states

def graph_exists(dir_, name):
    if not os.path.exists(dir_):
        os.mkdir(dir_) 
    path = dir_ + name
    return os.path.exists(path)
    
import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm[0]

def get_experiments_channels(noise_model, m, instruction_set):
    if instruction_set == 0:
        channelX0 = noise_model.get_instruction_channel(m[0], Instruction.X)
        channelP2 = noise_model.get_meas_channel(Instruction.MEAS, m[2])
        channelCX02 = noise_model.get_instruction_channel(m[2], Instruction.CNOT, control=m[0])
        channelCX12 = noise_model.get_instruction_channel(m[2], Instruction.CNOT, control=m[1])
        return [channelCX02, channelCX12, channelP2, channelX0]
    else:
        assert instruction_set == 1
        channelH2 = noise_model.get_instruction_channel(m[2], Instruction.H)
        channelH1 = noise_model.get_instruction_channel(m[1], Instruction.H)
        channelP2 = noise_model.get_meas_channel(Instruction.MEAS, m[2])
        channelCX21 = noise_model.get_instruction_channel(m[1], Instruction.CNOT, control=m[2])
        channelCX01 = noise_model.get_instruction_channel(m[1], Instruction.CNOT, control=m[0])
        return [channelH2, channelH1, channelCX21, channelCX01, channelP2]

def get_guards(instruction_set, noise_model, m):
    if instruction_set == 0:
        return []
    else:
        channelP2 = noise_model.get_meas_channel(Instruction.MEAS, m[2])
        return [Guard(are_controls_entangled, [channelP2])]

    
def get_all_experiments_channels(noise_model, m):
    channelX0 = noise_model.get_instruction_channel(m[0], Instruction.X)
    channelP2 = noise_model.get_meas_channel(Instruction.MEAS, m[2])
    channelCX02 = noise_model.get_instruction_channel(m[2], Instruction.CNOT, control=m[0])
    channelCX12 = noise_model.get_instruction_channel(m[2], Instruction.CNOT, control=m[1])

    channelH2 = noise_model.get_instruction_channel(m[2], Instruction.H)
    channelH1 = noise_model.get_instruction_channel(m[1], Instruction.H)
    channelCX21 = noise_model.get_instruction_channel(m[1], Instruction.CNOT, control=m[2])
    channelCX01 = noise_model.get_instruction_channel(m[1], Instruction.CNOT, control=m[0])

    return [channelX0, channelP2, channelCX02, channelCX12, channelH2, channelH1, channelCX21, channelCX01]

def run_experiment_bellman(noise_model, m, graphs_path="", with_safety=False, instruction_set=1, experiment_name="bitflip"):
            
    sys.setrecursionlimit(6000)
    if not os.path.exists(graphs_path):
        os.mkdir(graphs_path) 
    file = open(f"exp_out/{experiment_name}" + ".csv", "w")
    file.write("horizon,lambda,time,channels\n")
    for horizon in range(BOTTOM_HORIZON, MAX_HORIZON):
        channels = get_experiments_channels(noise_model, m, instruction_set)
        # initial_states= get_initial_states(m)
        initial_belief, initial_states = get_belief_and_initial_states(m)    
        time_to_build = -1
        start = time.time()
        try :
            # with time_limit(3600*4):
            g = Graph(channels=channels, initial_states=initial_states, target_belief_threshold=1.0, initial_belief=initial_belief)
            g.build_graph(is_target_qs, m, obs_limit=1000000000, horizon=horizon, belief_threshold=0.0, max_layer_width=10000000000, do_safety_check=with_safety, debug=False)
            expected_val = get_bellman_graph(g, is_target_qs, address_space=m, outpath = f"{graphs_path}bf_{horizon}.py", horizon=horizon)
            end = time.time()
            time_to_build = round(end - start,3)
            assert time_to_build != -1
        except TimeoutException as _:
            print("timeout:", time.time() - start)
            expected_val = -1
            time_to_build = -1    
        line = f"{horizon},{expected_val},{time_to_build},{len(channels)}\n"
        file.write(line)
        file.flush()
    file.close()


# traditional algorithm
def traditional_algorithm(meas_count, noise_model, seed, address_space):

    state = get_bell_state(random.randint(0, 3), address_space)
    qpu = QSimulator(noise_model, seed=seed)
    qpu.qmemory = state
    qpu.cx(address_space[0],address_space[2])
    qpu.cx(address_space[1],address_space[2])
    ins = 2
    count_ones = 0
    i = 0
    while i < meas_count:
        outcome = qpu.apply_instructions([Instruction.MEAS], [address_space[2]], [None])
        ins += 1
        count_ones += outcome
        i += 1
    if count_ones >= ceil(meas_count/2):
        qpu.apply_instructions([Instruction.X], [address_space[0]], [None])
        ins +=1
    if is_target_qs(qpu.qmemory, address_space):
        return 1, ins
    
    return 0, ins

def get_lambdas(backend, embedding_index, dir_prefix=""):
    f = open(dir_prefix + f"{backend}.txt")
    result = dict()
    for line in f.readlines()[1:]:
        elements = line.split(",")
        embedding_index_ = int(elements[0])
        horizon = int(elements[1])
        lambda_ = float(elements[2])

        if embedding_index == embedding_index_:
            assert horizon not in result.keys()
            result[horizon] = lambda_
    f.close()
    return result

def load_embedding(backend, embedding_index, dir_prefix=DIR_PREFIX, is_one=False):
    if is_one:
        f = open(dir_prefix + f"inverse_mappings1/{backend}_{embedding_index}.txt")
    else:
        f = open(dir_prefix + f"inverse_mappings/{backend}_{embedding_index}.txt")
    lines = f.readlines()
    result = dict()
    assert(len(lines) == 3)
    for line in lines:
        elements = line.split(' ')
        result[int(elements[1])] = int(elements[0])
    f.close()
    return result

def get_num_embeddings(backend, dir_prefix):
    f = open(dir_prefix + f"{backend}.txt")
    all_embeddings = set()
    for line in f.readlines()[1:]:
        elements = line.split(",")
        embedding_index = int(elements[0])
        all_embeddings.add(embedding_index)
    f.close()
    return len(all_embeddings)
    
def test_programs(shots=2000, for_ibm=False, factor=2):
    output_file = open(DIR_PREFIX + "analysis_results/test_lambdas.csv", "w")
    output_file.write("backend,horizon,lambda,acc,diff\n")
    
    for backend in backends_w_embs: 
        num_embeddings = get_num_embeddings(backend, DIR_PREFIX + "lambdas/")
        noise_model = get_ibm_noise_model(backend)
        for index in range(num_embeddings):
            m = load_embedding(backend, index, DIR_PREFIX)
            lambdas_d = get_lambdas(backend, index, DIR_PREFIX + "lambdas/") 
            for horizon in range(4,8):
                
                algorithms = load_algorithms_file(DIR_PREFIX + f'algorithms/{backend}_{index}_{horizon}')
                assert len(algorithms) == 1
                for algorithm in algorithms[:3]:
                    if for_ibm:
                        acc = 0
                        for _ in range(factor):
                            acc += ibm_execute_my_algo(shots, algorithm, backend, m)
                        acc /= factor  
                    else:
                        acc, _ = execute_my_algo(shots, algorithm, noise_model,
                                                    is_target_qs, address_space=m)
                    acc = round(acc, 3)
                    output_file.write(f"{backend}-{index},{horizon},{lambdas_d[horizon]},{acc},{round(lambdas_d[horizon]-acc,3)}\n")
        output_file.flush()
                    
    output_file.close()

def test_programs1(shots=2000, for_ibm=False, factor=2):
    output_file = open(DIR_PREFIX + "analysis_results1/test_lambdas.csv", "w")
    output_file.write("backend,horizon,lambda,acc,diff\n")
    
    for backend in backends_w_embs: 
        num_embeddings = get_num_embeddings(backend, DIR_PREFIX + "lambdas1/")
        noise_model = get_ibm_noise_model(backend)
        for index in range(num_embeddings):
            m = load_embedding(backend, index, DIR_PREFIX, is_one=True)
            lambdas_d = get_lambdas(backend, index, DIR_PREFIX + "lambdas1/") 
            for horizon in range(4,8):
                
                algorithms = load_algorithms_file(DIR_PREFIX + f'algorithms1/{backend}_{index}_{horizon}')
                for algorithm in algorithms[:3]:
                    if for_ibm:
                        acc = 0
                        for _ in range(factor):
                            acc += ibm_execute_my_algo(shots, algorithm, backend, m)
                        acc /= factor  
                    else:
                        acc, _ = execute_my_algo(shots, algorithm, noise_model,
                                                    is_target_qs, address_space=m)
                    acc = round(acc, 3)
                    output_file.write(f"{backend}-{index},{horizon},{lambdas_d[horizon]},{acc},{round(lambdas_d[horizon]-acc,3)}\n")
        output_file.flush()
                    
    output_file.close()

def bellman_hardware_experiments(backend: str, do_safety_check=False, experiment_name="bitflip", instruction_set=1, mappings=None):    
    assert mappings is not None
    for b in [True, False]:
        Precision.is_lowerbound = b
        noise_model = get_ibm_noise_model(backend)
        for (index, embedding) in enumerate(mappings):
            if instruction_set == 0:
                new_embedding = embedding
            else:
                assert instruction_set == 1
                new_embedding = dict()
                new_embedding[0] = embedding[0]
                new_embedding[1] = embedding[2]
                new_embedding[2] = embedding[1]
                
            run_experiment_bellman(noise_model, new_embedding, graphs_path=f"exp_out/bf_{backend}{Precision.is_lowerbound}_graphs{index}/", with_safety=do_safety_check, instruction_set=instruction_set, experiment_name=f"{backend}{Precision.is_lowerbound}_{experiment_name}{index}")

def simulator_experiments(shots=5000, for_ibm=False, is_one=False, factor=1):
    assert isinstance(for_ibm, bool)  
    if is_one: 
        output_file = open(DIR_PREFIX + "analysis_results1/backends_vs.csv", "w")
    else:
        output_file = open(DIR_PREFIX + "analysis_results/backends_vs.csv", "w")
    output_file.write(f"horizon,diff_index,real_hardware,acc,ins\n")
    if is_one:
        bottom_h = 7
    else:
        bottom_h = 4
    for horizon in range(bottom_h, 8):
        if is_one:
            algorithms = load_algorithms_file(DIR_PREFIX + f'analysis_results1/diff{horizon}.py')
        else:
            algorithms = load_algorithms_file(DIR_PREFIX + f'analysis_results/diff{horizon}.py')
        for real_hardware in backends_w_embs:
            num_embeddings = get_num_embeddings(real_hardware, DIR_PREFIX + "lambdas/")
            noise_model = get_ibm_noise_model(real_hardware)
            for index2 in range(num_embeddings):
                real_m = load_embedding(real_hardware, index2, DIR_PREFIX, is_one)
                for (index, algorithm) in enumerate(algorithms):
                    if for_ibm:
                        ins = -1
                        acc = 0
                        for _ in range(factor):
                            acc += ibm_execute_my_algo(shots, algorithm, real_hardware, real_m)
                        acc /= factor
                    else:
                        acc, ins = execute_my_algo(shots, algorithm, noise_model, is_target_qs, address_space=real_m)
                    acc = round(acc, 3)
                    output_file.write(f"{horizon},{index},{real_hardware}{index2},{acc},{ins}\n")
            output_file.flush()
    output_file.close()
        

def generate_pomdps():
    def is_target_qs2(hybrid_state):
        qs, cs = hybrid_state
        return is_target_qs(qs, m)

    for backend in backends_w_embs:
        embeddings = get_backend_embeddings(backend)
        for (index, m) in enumerate(embeddings):
            print(backend, index, m)
            initial_state = (QuantumState(0), ClassicalState())
            noise_model = get_ibm_noise_model(backend)
            channels = get_experiments_channels(noise_model, m, 0)
            initial_distribution = []
            for s in get_initial_states(m):
                initial_distribution.append((s, 0.25))
            pomdp = build_pomdp(channels, initial_state, m, initial_distribution)
            pomdp.serialize(is_target_qs2, f"{DIR_PREFIX}pomdps/{backend}_{index}.txt")
            f = open(f"{DIR_PREFIX}inverse_mappings/{backend}_{index}.txt", "w")
            for i in range(3):
                f.write(f"{m[i]} {i}\n")
            f.close()

def times_generate_pomdps():
    times_file = open(f"{DIR_PREFIX}analysis_results/pomdp_times.csv", "w")
    times_file.write("backend,embedding,time\n")

    for backend in backends_w_embs:
        embeddings = get_backend_embeddings(backend)
        for (index, m) in enumerate(embeddings):
            print(backend, index, m)
            start_time = time.time()
            initial_state = (QuantumState(0), ClassicalState())
            noise_model = get_ibm_noise_model(backend)
            channels = get_experiments_channels(noise_model, m, 0)
            initial_distribution = []
            for s in get_initial_states(m):
                initial_distribution.append((s, 0.25))
            pomdp = build_pomdp(channels, initial_state, m, initial_distribution)
            end_time = time.time()
            times_file.write(f"{backend},{index},{end_time-start_time}\n")
        times_file.flush()
    times_file.close()

            

def generate_pomdps1(arg):
    backend = backends_w_embs[arg]
    times_file = open(f"{DIR_PREFIX}analysis_results1/pomdp_times_{backend}.csv", "w")
    # times_file.write("backend,embedding,time\n")
    def is_target_qs2(hybrid_state):
        qs, cs = hybrid_state
        return is_target_qs(qs, embedding)
    # for backend in backends_w_embs:
    num_embeddings = get_num_embeddings(backend, DIR_PREFIX + "lambdas/")
    for embedding_index in range(num_embeddings):
        print(backend, embedding_index)
        embedding_ = load_embedding(backend, embedding_index, DIR_PREFIX)
        embedding = dict()
        embedding[0] = embedding_[0]
        embedding[1] = embedding_[2]
        embedding[2] = embedding_[1]
        start_time = time.time()
        f_new_embedding = open(DIR_PREFIX + f"inverse_mappings1/{backend}_{embedding_index}.txt", "w")
        for  i in range(3):
            f_new_embedding.write(f"{embedding[i]} {i}\n")
        f_new_embedding.close()
        start_time = time.time()
        initial_state = (QuantumState(0), ClassicalState())
        noise_model = get_ibm_noise_model(backend)
        channels = get_experiments_channels(noise_model, embedding, 1)
        initial_distribution = []
        for s in get_initial_states(embedding):
            initial_distribution.append((s, 0.25))
        guards = get_guards(1, noise_model, embedding)
        pomdp = build_pomdp(channels, initial_state, embedding, initial_distribution, guards)
        end_time = time.time()
        times_file.write(f"{backend},{embedding_index},{end_time-start_time}\n")
        times_file.flush()
        pomdp.serialize(is_target_qs2, DIR_PREFIX + f"pomdps1/{backend}_{embedding_index}.txt")
    times_file.close()

def times_generate_pomdps1():
    times_file = open(f"{DIR_PREFIX}analysis_results1/pomdp_times.csv", "w")
    times_file.write("backend,embedding,time\n")
    for backend in backends_w_embs:
        backend = backends_w_embs[arg]
        num_embeddings = get_num_embeddings(backend, DIR_PREFIX + "lambdas/")
        for embedding_index in range(num_embeddings):
            print(backend, embedding_index)
            embedding_ = load_embedding(backend, embedding_index, DIR_PREFIX)
            embedding = dict()
            embedding[0] = embedding_[0]
            embedding[1] = embedding_[2]
            embedding[2] = embedding_[1]
            start_time = time.time()
            initial_state = (QuantumState(0), ClassicalState())
            noise_model = get_ibm_noise_model(backend)
            channels = get_experiments_channels(noise_model, embedding, 1)
            initial_distribution = []
            for s in get_initial_states(embedding):
                initial_distribution.append((s, 0.25))
            guards = get_guards(1, noise_model, embedding)
            pomdp = build_pomdp(channels, initial_state, embedding, initial_distribution, guards)
            end_time = time.time()
            times_file.write(f"{backend},{embedding_index},{end_time-start_time}\n")
        times_file.flush()
    times_file.close()



def generate_backends_to_num_embeddings():
    result = "{"
    for (index, backend) in enumerate(backends_w_embs):
        result+="{"
        result+= f"\"{backend}\",{len(get_backend_embeddings(backend))}"
        if index +1 == len(backends_w_embs):
            result+="}\n"
        else:
            result+="},\n"
    result+="}"
    print(result)

def generate_all_experiments_script():
    f = open("run_all_experiments.sh", "w")
    for backend in backends_w_embs:
        f.write(f"sbatch experiments_script.sh ./inputs/{backend}.input\n")
    f.close()

def generate_inputs_file():
    for backend in backends_w_embs:
        f = open(f"{DIR_PREFIX}inputs/{backend}.input", "w")
        temp_backend = backend.replace("fake_", "")
        f.write(f"{temp_backend}\n")
        f.close()

if __name__ == "__main__":
    arg_backend = sys.argv[1]
    
    Precision.PRECISION = 7
    Precision.update_threshold()
    assert arg_backend in [ "testprograms","ibmtestprograms", "ibmsimulatorexp1", "simulatorexp", "ibmsimulatorexp", "genpomdp", "genpomdp1", "timesgenpomdp", "timesgenpomdp1", "ibmtestprograms1"]
    
    if arg_backend == "genpomdp":
        Precision.PRECISION = 10
        Precision.update_threshold()
        generate_pomdps()

    if arg_backend == "timesgenpomdp":
        Precision.PRECISION = 10
        Precision.update_threshold()
        times_generate_pomdps()

    if arg_backend == "genpomdp1":
        arg2 = int(sys.argv[2])
        Precision.PRECISION = 10
        Precision.update_threshold()
        generate_pomdps1(arg2)

    if arg_backend == "timesgenpomdp1":
        Precision.PRECISION = 10
        Precision.update_threshold()
        times_generate_pomdps1()

    if arg_backend == "testprograms":
        Precision.PRECISION = 30
        Precision.update_threshold()
        test_programs("exp_out/allprogramstests.csv", shots=5000,for_ibm=False)

    if arg_backend == "ibmtestprograms":
        test_programs(shots=5000,for_ibm=True)

    if arg_backend == "ibmtestprograms1":
        test_programs1(shots=5000,for_ibm=True)

    if arg_backend == "simulatorexp":
        Precision.PRECISION = 30
        Precision.update_threshold()
        simulator_experiments("exp_out/simulators.csv",2000, for_ibm=False)

    if arg_backend == "ibmsimulatorexp":
        simulator_experiments(5000, for_ibm=True, factor=3)

    if arg_backend == "ibmsimulatorexp1":
        simulator_experiments(5000, for_ibm=True, is_one=True, factor=2)
    
