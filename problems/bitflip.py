import os, sys
sys.path.append(os.getcwd()+"/..")

from cmath import isclose
from copy import deepcopy
import time
from typing import Dict, List
import json

from qiskit import IBMQ, ClassicalRegister, QuantumCircuit, QuantumRegister, execute
from algorithm import AlgorithmNode, execute_algorithm, load_algorithms_file
from cmemory import ClassicalState
from pomdp import POMDP, POMDPAction, POMDPVertex, build_pomdp
import qmemory
from qpu_utils import GateData, Op, BasisGates
from utils import are_matrices_equal, is_matrix_in_list, Precision
sys.path.append(os.getcwd()+"/..")

from qstates import QuantumState
from ibm_noise_models import Instruction, MeasChannel, NoiseModel, get_ibm_noise_model, HardwareSpec, ibm_simulate_circuit
import numpy as np
from math import pi   
from enum import Enum
from experiments_utils import ReadoutNoise




POMDP_OUTPUT_DIR = "/Users/stefaniemuroyalei/Documents/ist/im_time_evolution/synthesis/bitflip/"
WITH_TERMALIZATION = False
MAX_HORIZON = 7
MIN_HORIZON = 4
MAX_PRECISION = 10
TIME_OUT = 10800 # (in seconds) i.e 3 hours

EMBEDDINGS_FILE = "embeddings.json"

bell0_real_rho = [
                    [0.5, 0, 0, 0.5],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0.5, 0, 0, 0.5],
                ]
        
bell1_real_rho = [
                    [0, 0, 0, 0],
                    [0, 0.5, 0.5, 0],
                    [0, 0.5, 0.5, 0],
                    [0, 0, 0, 0],
                ]
        
bell2_real_rho = [
                    [0.5, 0, 0, -0.5],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [-0.5, 0, 0, 0.5],
                ]
        
bell3_real_rho = [
                    [0, 0, 0, 0],
                    [0, 0.5, -0.5, 0],
                    [0, -0.5, 0.5, 0],
                    [0, 0, 0, 0],
                ]
        
bell_state_pts = [bell0_real_rho, bell1_real_rho, bell2_real_rho, bell3_real_rho]

class ExperimentID(Enum):
    IPMA = 0
    CXH = 1


class BitFlipInstance:
    def __init__(self,embedding: Dict[int, int]):
        """_summary_

        Args:
            num_qubits (int): _description_
            instruction_set (List[Instruction]): _description_
            embedding (Dict[int, int]): a mapping from logical qubits to physical qubits
        """
        self.embedding = embedding
        self.num_qubits = max(embedding.values())
        self.initial_state = None
        self.get_initial_states()
        # check embedding
        assert 0 in self.embedding.keys()
        assert 1 in self.embedding.keys()
        assert 2 in self.embedding.keys()
        assert len(self.embedding.keys()) == 3

    def get_initial_states(self):
        """
        The initial state is specified as a uniform superpositions over all four bell states.
        """
        self.initial_state = []
        initial_cs = ClassicalState()

        H0 = Instruction(self.embedding[0], Op.H).get_gate_data()
        CX01 = Instruction(self.embedding[1], Op.CNOT, self.embedding[0]).get_gate_data()
        X0 = Instruction(self.embedding[0], Op.X).get_gate_data()
        Z0 = Instruction(self.embedding[0], Op.Z).get_gate_data()

        # prepare first bell state
        bell0 = QuantumState(0, qubits_used=list(self.embedding.values()))
        bell0 = qmemory.handle_write(bell0, H0)
        bell0 = qmemory.handle_write(bell0, CX01)
        self.initial_state.append((bell0, initial_cs))

        # prepare second bell state
        bell1 = qmemory.handle_write(bell0, X0)
        self.initial_state.append((bell1, initial_cs))
    
        # prepare third bell state
        bell2 = qmemory.handle_write(bell0, Z0)
        self.initial_state.append((bell2, initial_cs))

        # preapre fourth bell state
        bell3 = qmemory.handle_write(bell2, X0)
        self.initial_state.append((bell3, initial_cs))


    def is_target_qs(self, hybrid_state) -> bool:
        qs , _ = hybrid_state
        current_rho = qs.single_partial_trace(index=self.embedding[2])
        initial_qs, _ = self.initial_state[0]
        bell0_rho = initial_qs.single_partial_trace(index=self.embedding[2])
        initial_qs,_ = self.initial_state[2]
        bell1_rho = initial_qs.single_partial_trace(index=self.embedding[2])
        assert len(bell0_rho) == 4
        assert len(bell1_rho) == 4
        return are_matrices_equal(current_rho, bell0_rho) or are_matrices_equal(current_rho, bell1_rho)
    

# choosing embeddings
def get_pivot_qubits(noise_model: NoiseModel):
    result = set()
    noises = []
    if noise_model.hardware_spec == HardwareSpec.MELBOURNE:
        noise_model.num_qubits = 14
    for qubit in range(noise_model.num_qubits):
        if noise_model.get_qubit_indegree(qubit) > 1:
            noise_data = noise_model.instructions_to_channel[Instruction(qubit, Op.MEAS)]
            assert isinstance(noise_data, MeasChannel)
            success0 = noise_data.get_ind_probability(0,0)
            success1 = noise_data.get_ind_probability(1,1)
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

def get_selected_couplers(noise_model, target):
    couplers = noise_model.get_qubit_couplers(target)
    first_pair = (couplers[0], couplers[1]) # most noisy pair of couplers for this target
    second_pair = (couplers[len(couplers) -1], couplers[len(couplers) -2]) # least noisy pair of couplers for this target
    return [first_pair, second_pair]

def does_result_contains_d(result, d):
    for d_ in result:
        controls1 = set([d[0], d[1]])
        controls2 = set([d_[0], d_[1]])
        if d_[2] == d[2] and controls1 == controls2:
            return True
    return False

def get_backend_embeddings(backend: HardwareSpec):
    result = []
    noise_model = NoiseModel(backend, thermal_relaxation=WITH_TERMALIZATION)
    if noise_model.num_qubits < 14:
        pivot_qubits = set()
        for qubit in range(noise_model.num_qubits):
            if noise_model. get_qubit_indegree(qubit) > 1:
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

class IBMBitFlipInstance:
    SHOTS = 1024 * 4
    def __init__(self, embedding) -> None:
        self.embedding = embedding
        self.bitflip_instance = BitFlipInstance(embedding)

    def prepare_bell_state(qc: QuantumCircuit, bell_index):
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

    def is_target_qs(self, state_vector):
        assert len(state_vector) == 8 # 3 qubits
        qs = None
        for (index, amp) in state_vector:
            if not isclose(amp, 0.0, abs_tol=Precision.isclose_abstol):
                if qs is None:
                    qs = QuantumState(index, amp, qubits_used=list(self.embedding.values()))
                else:
                    assert qs.get_amplitude(index) == 0.0
                    qs.insert_amplitude(index, amp) 

        return self.bitflip_instance.is_target_qs(qs)
    
    def ibm_execute_my_algo(self, shots, alg, backend, is_simulated=True, log=None):
        shots_per_initial_state = shots/4
        accuracy = 0
        initial_layout = [self.embedding[0], self.embedding[1], self.embedding[2]]
        ibm_noise_model = get_ibm_noise_model(backend, thermal_relaxation=WITH_TERMALIZATION)
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
            self.prepare_bell_state(qc, bell_state_index)
            if isinstance(alg, AlgorithmNode):
                execute_algorithm(alg, qc, basis_gates=basis_gates, cbits=cr)
            else:
                alg(qc, basis_gates, cr)
            if is_simulated:
                qc.save_statevector('res', pershot=True)
                state_vs = ibm_simulate_circuit(qc, ibm_noise_model, shots_per_initial_state, initial_layout)
                for state in state_vs:
                    if self.is_target_qs(state):
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
    
def is_hardware_selected(noise_model: NoiseModel):
    return (Op.CNOT in noise_model.basis_gates.value) and len(noise_model.instructions_to_channel.keys()) > 0
        

def generate_embeddings():
    if not os.path.exists(POMDP_OUTPUT_DIR):
        os.mkdir(POMDP_OUTPUT_DIR) 
        
    result = dict()
    c_embeddings = 0
    for hardware_spec in HardwareSpec:
        noise_model = NoiseModel(hardware_spec, thermal_relaxation=WITH_TERMALIZATION)
        if is_hardware_selected(noise_model):
            assert hardware_spec not in result.keys()
            result[hardware_spec.value] = dict()
            embeddings = get_backend_embeddings(hardware_spec)
            result[hardware_spec.value]["count"] = len(embeddings)
            result[hardware_spec.value]["embeddings"] = embeddings
            c_embeddings += len(embeddings)

    result["count"] = c_embeddings
    f = open(f"{POMDP_OUTPUT_DIR}{EMBEDDINGS_FILE}", "w")
    f.write(json.dumps(result))
    f.close()

def load_embeddings(experiment_id: ExperimentID):
    with open(f"{POMDP_OUTPUT_DIR}{EMBEDDINGS_FILE}", 'r') as file:
        result = dict()
        data = json.load(file)
        assert data["count"] == 225
        result["count"] = 225

        for hardware_spec in HardwareSpec:
            noise_model = NoiseModel(hardware_spec, thermal_relaxation=WITH_TERMALIZATION)
            if is_hardware_selected(noise_model):
                result[hardware_spec] = dict()
                result[hardware_spec]["count"] = data[hardware_spec.value]["count"]
                result[hardware_spec]["embeddings"] = []

                for embedding in data[hardware_spec.value]["embeddings"]:
                    d = dict()
                    for (key, value) in embedding.items():
                        d[int(key)] = int(value)
                    if experiment_id == ExperimentID.CXH:
                        temp = d[2]
                        d[2] = d[1]
                        d[1] = temp
                    result[hardware_spec]["embeddings"].append(d)
            else:
                assert hardware_spec.value not in data.keys()
        
        return result
    raise Exception(f"could not load embeddings file {POMDP_OUTPUT_DIR}{EMBEDDINGS_FILE}")

def get_experiments_actions(noise_model: NoiseModel, embedding: Dict[int,int], experiment_id: ExperimentID):
    if experiment_id == ExperimentID.IPMA:
        if noise_model.basis_gates in [BasisGates.TYPE1]:
            X0 = POMDPAction("X0", [Instruction(embedding[0], Op.U3, params=[pi, 2*pi, pi])])
        else:
            X0 = POMDPAction("X0", [Instruction(embedding[0], Op.X)])
            
        P2 = POMDPAction("P2", [Instruction(embedding[2], Op.MEAS)])
        CX02 = POMDPAction("CX02", [Instruction(embedding[2], Op.CNOT, control=embedding[0])])
        CX12 = POMDPAction("CX12", [Instruction(embedding[2], Op.CNOT, control=embedding[1])])
        return [CX02, CX12, P2, X0]
    else:
        assert experiment_id == ExperimentID.CXH
        if noise_model.basis_gates in [BasisGates.TYPE1, BasisGates.TYPE6]:
            H2 = POMDPAction("H2", [Instruction(embedding[2], Op.U2, params=[0, pi])])
            H1 = POMDPAction("H1", [Instruction(embedding[1], Op.U2, params=[0, pi])])
        else:
            assert noise_model.basis_gates in [BasisGates.TYPE2, BasisGates.TYPE3, BasisGates.TYPE4, BasisGates.TYPE7]
            H2 = POMDPAction("H2", [
                Instruction(embedding[2], Op.RZ, params=[pi/2]),
                Instruction(embedding[2], Op.SX),
                Instruction(embedding[2], Op.RZ, params=[pi/2])
            ])
            H1 = POMDPAction("H1", [
                Instruction(embedding[1], Op.RZ, params=[pi/2]),
                Instruction(embedding[1], Op.SX),
                Instruction(embedding[1], Op.RZ, params=[pi/2])
            ])
        
        P2 = POMDPAction("P2", [Instruction(embedding[2], Op.MEAS)])
        CX21 = POMDPAction("CX21", [Instruction(embedding[1], Op.CNOT, control=embedding[2])])
        CX01 = POMDPAction("CX01", [Instruction(embedding[1], Op.CNOT, control=embedding[0])])
        return [H2, H1, CX21, CX01, P2]
    

def guard(vertex: POMDPVertex, embedding: Dict[int, int], action: POMDPAction):
    assert len(action.instruction_sequence) == 1
    if action.instruction_sequence[0].op != Op.MEAS:
        return True
    assert isinstance(vertex, POMDPVertex)
    assert isinstance(embedding, dict)
    assert isinstance(action, POMDPAction)
    qs = vertex.quantum_state
    meas_instruction = Instruction(embedding[2], Op.MEAS)
    qs0 = qmemory.handle_write(qs, meas_instruction.get_gate_data(is_meas_0=True))
    qs1 = qmemory.handle_write(qs, meas_instruction.get_gate_data(is_meas_0=False))

    if qs0 is not None:
        pt0 = qs0.single_partial_trace(index=embedding[2])
        if not is_matrix_in_list(pt0, bell_state_pts):
            return False
    
    if qs1 is not None:
        pt1 = qs1.single_partial_trace(index=embedding[2])
        return is_matrix_in_list(pt1, bell_state_pts)
    return True
    

def generate_pomdp(experiment_id: ExperimentID, hardware_spec: HardwareSpec, 
                embedding: Dict[int, int], pomdp_write_path: str, return_pomdp=False):
    noise_model = NoiseModel(hardware_spec, thermal_relaxation=WITH_TERMALIZATION)
    if not is_hardware_selected(noise_model):
        return None
    bitflip_instance = BitFlipInstance(embedding)
    actions = get_experiments_actions(noise_model, embedding, experiment_id)
    initial_distribution = []
    for s in bitflip_instance.initial_state:
        initial_distribution.append((s, 0.25))

    start_time = time.time()
    pomdp = build_pomdp(actions, noise_model, MAX_HORIZON, embedding, initial_distribution=initial_distribution, guard=guard)
    end_time = time.time()
    if return_pomdp:
        return pomdp
    pomdp.serialize(bitflip_instance, pomdp_write_path)
    return end_time-start_time
    
def generate_pomdps(experiment_id: ExperimentID):
    if not os.path.isdir(f"{POMDP_OUTPUT_DIR}pomdps{experiment_id.value}/"):
        os.mkdir(f"{POMDP_OUTPUT_DIR}pomdps{experiment_id.value}/")
    
    if not os.path.isdir(f"{POMDP_OUTPUT_DIR}analysis_results{experiment_id.value}/"):
        os.mkdir(f"{POMDP_OUTPUT_DIR}analysis_results{experiment_id.value}/")

    all_embeddings = load_embeddings(experiment_id)
    times_file = open(f"{POMDP_OUTPUT_DIR}analysis_results{experiment_id.value}/pomdp_times.csv", "w")
    times_file.write("backend,embedding,time\n")
    for backend in HardwareSpec:
        try:
            embeddings = all_embeddings[backend]["embeddings"]
            for (index, m) in enumerate(embeddings):
                print(backend, index, m)
                time_taken = generate_pomdp(experiment_id, backend, m, f"{POMDP_OUTPUT_DIR}pomdps{experiment_id.value}/{backend.value}_{index}.txt")
                if time_taken is not None:
                    times_file.write(f"{backend.name},{index},{time_taken}\n")
            times_file.flush()
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            
    times_file.close()

def get_lambdas(backend, embedding_index, dir_prefix=""):
    f = open(dir_prefix + f"{backend.value}.txt")
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

def test_programs(experiment_id: ExperimentID, shots=2000, factor=2):
    if not os.path.exists(POMDP_OUTPUT_DIR + "analysis_results{experiment_id.value}/"):
        os.mkdir(POMDP_OUTPUT_DIR + "analysis_results{experiment_id.value}/") 
    
    if not os.path.exists(POMDP_OUTPUT_DIR + "lambdas{experiment_id.value}/"):
        raise Exception(f"Guarantees not computed yet (directory lambdas{experiment_id.value}/ does not exists)")
    
    if not os.path.exists(POMDP_OUTPUT_DIR + f"algorithms{experiment_id.value}/"):
        raise Exception(f"Optimal algorithms not computed yet (directory algorithms{experiment_id.value}/ does not exists)")
    
    output_file = open(POMDP_OUTPUT_DIR + f"analysis_results{experiment_id.value}/test_lambdas.csv", "w")
    output_file.write("backend,horizon,lambda,acc,diff\n")
    
    all_embeddings = load_embeddings(experiment_id)
    for backend in HardwareSpec: 
        embeddings = all_embeddings[backend]["embeddings"]
        for (index, embedding) in enumerate(embeddings):
            if experiment_id == ExperimentID.IPMA:
                m = embedding
            else:
                m = dict()
                m[0] = embedding[0]
                m[1] = embedding[2]
                m[2] = embedding[1]
            ibm_bitflip_instance = IBMBitFlipInstance(m)
            lambdas_d = get_lambdas(backend, index, POMDP_OUTPUT_DIR + f"lambdas{experiment_id.value}/") 
            for horizon in range(MIN_HORIZON, MAX_HORIZON):
                algorithms = load_algorithms_file(POMDP_OUTPUT_DIR + f'algorithms{experiment_id.value}/{backend}_{index}_{horizon}')
                assert len(algorithms) == 1
                for algorithm in algorithms[:3]:
                    acc = 0
                    for _ in range(factor):
                        acc += ibm_bitflip_instance.ibm_execute_my_algo(shots, algorithm, backend)
                    acc /= factor  
                    acc = round(acc, 3)
                    output_file.write(f"{backend}-{index},{horizon},{lambdas_d[horizon]},{acc},{round(lambdas_d[horizon]-acc,3)}\n")
        output_file.flush()
    output_file.close()

class Test:
    real_embedding_count = {"fake_athens":3, "fake_belem":3, "fake_tenerife":2, "fake_lima":3, "fake_rome":3, "fake_manila":3, "fake_santiago":3, "fake_bogota":3, "fake_ourense":3,
        "fake_yorktown":6, "fake_essex":3, "fake_vigo":3, "fake_burlington":3, "fake_quito":3,
        "fake_london":3, "fake_jakarta":5, "fake_oslo":5, "fake_perth":5, "fake_lagos":5,"fake_nairobi":5, "fake_casablanca":5, "fake_melbourne":8, "fake_guadalupe":4,"fake_tokyo":10, "fake_poughkeepsie":6, "fake_johannesburg":5, "fake_boeblingen":7, "fake_almaden":9, "fake_singapore":6, "fake_mumbai":8, "fake_paris":7, "fake_auckland":2,"fake_kolkata":7, "fake_toronto":9, "fake_montreal":4, "fake_sydney":6, "fake_cairo":10,"fake_hanoi":7, "fake_geneva":3, "fake_cambridge":5, "fake_rochester":5, "fake_brooklyn":6,"fake_manhattan":7, "fake_washington":7}

    @staticmethod
    def run_all():
        Test.check_bell_state_creation()

    @staticmethod
    def check_bell_state_creation():

        bf = BitFlipInstance({0:0, 1:1, 2:2})

        bf.get_initial_states()

        for (index, initial_state) in enumerate(bf.initial_state):
            qs, cs = initial_state
            assert isinstance(qs, QuantumState)
            current_rho = qs.single_partial_trace(index=2)
            assert are_matrices_equal(current_rho, bell_state_pts[index])

    @staticmethod
    def _load_real_embeddings(hardware: HardwareSpec) -> List[Dict[int, int]]:
        answer = []
        num_embeddings = Test.real_embedding_count[hardware.value]
        for index in range(num_embeddings):
            f = open(f"/Users/stefaniemuroyalei/Documents/ist/im_time_evolution/algorithm_synthesis/qalgorithm_synthesis/inverse_mappings/{hardware.value}_{index}.txt")

            lines = f.readlines()
            assert len(lines) == 3
            embedding = dict()
            for (index, line) in enumerate(lines):
                elements = line.split(" ")
                embedding[index] = int(elements[0])
            f.close()
            answer.append(embedding)
        return answer
    
    @staticmethod
    def get_diff_embeddings(current_embeddings, real_embeddings):
        in_current = []
        for embedding in current_embeddings:
            if embedding not in real_embeddings:
                in_current.append(embedding)
        
        in_real = []
        for embedding in real_embeddings:
            if embedding not in current_embeddings:
                in_real.append(embedding)
        return in_current, in_real

    @staticmethod
    def __check_embeddings(hardware_spec, embeddings):
        real_embeddings = Test._load_real_embeddings(hardware_spec)
        in_current, in_real = Test.get_diff_embeddings(embeddings, real_embeddings)

        if len(in_current) > 0 or len(in_real) > 0:
            print("Pivot Qubits: ")
            raise Exception(f"Embeddings do not match {hardware_spec}: Pivot Qubits: {get_pivot_qubits(NoiseModel(hardware_spec, thermal_relaxation=WITH_TERMALIZATION))}\n{in_current}\n{in_real}")
        if len(embeddings) != Test.real_embedding_count[hardware_spec.value]:
            raise Exception(f"{hardware_spec} embedding count does not match with expected ({len(embeddings)} != {Test.real_embedding_count[hardware_spec.value]})")

    @staticmethod
    def check_embeddings():
        assert len(Test.real_embedding_count.keys()) == 44
        
        for hardware_spec in HardwareSpec:
            noise_model = NoiseModel(hardware_spec, thermal_relaxation=WITH_TERMALIZATION)
            if is_hardware_selected(noise_model):
                embeddings = get_backend_embeddings(hardware_spec)
                Test.__check_embeddings(hardware_spec, embeddings)
                
    
    @staticmethod
    def check_embeddings_file():
        all_embeddings = load_embeddings(ExperimentID.IPMA)

        for hardware_spec in HardwareSpec:
            noise_model = NoiseModel(hardware_spec, thermal_relaxation=WITH_TERMALIZATION)
            if is_hardware_selected(noise_model):
                embeddings = all_embeddings[hardware_spec]["embeddings"]
                Test.__check_embeddings(hardware_spec, embeddings)

            
    @staticmethod
    def check_selected_hardware():
        c = 0
        for hardware_spec in HardwareSpec:
            noise_model = NoiseModel(hardware_spec, thermal_relaxation=WITH_TERMALIZATION)

            if is_hardware_selected(noise_model):
                c += 1
        assert c == 44

    @staticmethod
    def check_instruction_sets(experiment_id: ExperimentID):
        all_embeddings = load_embeddings(experiment_id)
        for hardware_spec in HardwareSpec:
            noise_model = NoiseModel(hardware_spec, thermal_relaxation=WITH_TERMALIZATION)
            if is_hardware_selected(noise_model):
                hardware_spec_embeddings = all_embeddings[hardware_spec]["embeddings"]
                for embedding in hardware_spec_embeddings:
                    actions = get_experiments_actions(noise_model, embedding, experiment_id)
                    for action in actions:
                        assert isinstance(action, POMDPAction)
                        for instruction in action.instruction_sequence:
                            assert isinstance(instruction, Instruction)
                            if instruction not in noise_model.instructions_to_channel.keys():
                                for instruction_ in noise_model.instructions_to_channel.keys():
                                    print(instruction_)
                                raise Exception(f"{instruction} not in noise model ({hardware_spec.value}) ({noise_model.basis_gates})")
                            
    @staticmethod
    def test_pomdp(hardware_spec, embedding_index, experiment_id):
        all_embeddings = load_embeddings(ExperimentID.IPMA)
        embedding = all_embeddings[hardware_spec]["embeddings"][embedding_index]
        
        time = generate_pomdp(experiment_id, hardware_spec, embedding, f"{POMDP_OUTPUT_DIR}pomdps{experiment_id.value}/{hardware_spec.value}_{embedding_index}.txt")

        print(f"time taken to build (seconds): {time}")
        
    @staticmethod
    def dump_actions(hardware_spec, embedding_index, experiment_id):
        noise_model = NoiseModel(hardware_spec, thermal_relaxation=WITH_TERMALIZATION)
        all_embeddings = load_embeddings(experiment_id)
        embedding = all_embeddings[hardware_spec]['embeddings'][embedding_index]
        actions = get_experiments_actions(noise_model, embedding, experiment_id)
        
        for action in actions:
            assert isinstance(action, POMDPAction)
            print(action.name, action.instruction_sequence)
            assert len(action.instruction_sequence) == 1
            target_instruction = action.instruction_sequence[0]
            print(noise_model.instructions_to_channel[target_instruction])
            print()
            
    @staticmethod
    def print_pomdp(hardware_spec, embedding_index, experiment_id):
        all_embeddings = load_embeddings(experiment_id)
        embedding = all_embeddings[hardware_spec]['embeddings'][embedding_index]
        pomdp = generate_pomdp(experiment_id, hardware_spec, embedding, "", return_pomdp=True)
        assert isinstance(pomdp, POMDP)
        pomdp.print_graph()
        
    @staticmethod
    def parse_lambdas_file(path):
        result = dict()
        f = open(path)
        lines = f.readlines()
        assert lines[0] == "embedding,horizon,lambda,time\n" # first line is the heading
        for line in lines[1:]:
            elements = line.split(",")
            embedding_index = int(elements[0])
            horizon = int(elements[1])
            lambda_ = float(elements[2])
            if embedding_index not in result.keys():
                result[embedding_index] = dict()
            assert horizon not in result[embedding_index].keys()
            result[embedding_index][horizon] = lambda_
        f.close()
    
    
    @staticmethod
    def compare_lambdas(hardware_spec):
        # this is only for experiment IPMA
        old_lambdas = Test.parse_lambdas_file(f"/Users/stefaniemuroyalei/Documents/ist/im_time_evolution/algorithm_synthesis/qalgorithm_synthesis/lambdas/{hardware_spec.value}.txt")

if __name__ == "__main__":
    arg_backend = sys.argv[1]
    Precision.PRECISION = MAX_PRECISION
    Precision.update_threshold()

    if arg_backend == "embeddings":
        # step 1
        generate_embeddings()   

    if arg_backend == "all_pomdps":
        # step 2: generate all pomdps
        experiment_id = sys.argv[2]
        assert experiment_id.lower() in ["ipma", "cxh"]
        if experiment_id.lower() == "ipma":
            experiment_id = ExperimentID.IPMA
        else:
            experiment_id = ExperimentID.CXH
        generate_pomdps(experiment_id)


    if arg_backend == "test" :
        pass
        # Test.check_selected_hardware()
        # Test.check_embeddings()
        # Test.check_embeddings_file()
        # Test.check_bell_state_creation()
        # Test.check_instruction_sets(ExperimentID.IPMA)
        # Test.check_instruction_sets(ExperimentID.CXH)
        # Test.test_pomdp(HardwareSpec.ALMADEN, 0, ExperimentID.IPMA)
        # Test.dump_actions(HardwareSpec.TENERIFE, 0, ExperimentID.IPMA)
        # Test.print_pomdp(HardwareSpec.TENERIFE, 0, ExperimentID.IPMA)
        Test.compare_lambdas(HardwareSpec.TENERIFE)
        