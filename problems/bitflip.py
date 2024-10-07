import os, sys
sys.path.append(os.getcwd()+"/..")

from cmath import isclose
from copy import deepcopy
import time
from typing import Dict, List
import json

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from algorithm import AlgorithmNode, execute_algorithm
from cmemory import ClassicalState
from pomdp import POMDP, POMDPAction, POMDPVertex, build_pomdp
import qmemory
from qpu_utils import GateData, Op, BasisGates
from utils import are_matrices_equal, get_index, is_matrix_in_list, Precision
sys.path.append(os.getcwd()+"/..")

from qstates import QuantumState
from ibm_noise_models import Instruction, MeasChannel, NoiseModel, get_ibm_noise_model, HardwareSpec, ibm_simulate_circuit, load_config_file
import numpy as np
from math import pi   
from enum import Enum
from experiments_utils import ReadoutNoise

WITH_TERMALIZATION = False
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

class BitflipExperimentID(Enum):
    IPMA = "ipma"
    CXH = "cxh"


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
        
        new_embedding = dict()
        values = sorted(self.embedding.values())
        for (key, value) in self.embedding.items():
            new_embedding[key] = get_index(value, values)
        self.bitflip_instance = BitFlipInstance(new_embedding)

    @staticmethod
    def prepare_bell_state(qc: QuantumCircuit, bell_index: int):
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
        for (index, amp) in enumerate(state_vector):
            if not isclose(amp, 0.0, abs_tol=Precision.isclose_abstol):
                if qs is None:
                    qs = QuantumState(index, amp, qubits_used=list(self.embedding.keys()))
                else:
                    assert qs.get_amplitude(index) == 0.0
                    qs.insert_amplitude(index, amp) 

        return self.bitflip_instance.is_target_qs((qs, None))
    
    def ibm_execute_my_algo(self, alg, backend, is_simulated=True, log=None):
        accuracy = 0
        
        ibm_noise_model = get_ibm_noise_model(backend, thermal_relaxation=WITH_TERMALIZATION)

        real_hardware_name = backend.value.lower().replace("fake", "ibm")
        
        if log is not None:
            log_file = open(log, "w")

        for bell_state_index in range(4):
            qr = QuantumRegister(3)
            cr = ClassicalRegister(3)
            qc = QuantumCircuit(qr, cr)
            initial_layout = {qr[0]: self.embedding[0], qr[1]:self.embedding[1], qr[2]:self.embedding[2]}
            IBMBitFlipInstance.prepare_bell_state(qc, bell_state_index)
            assert isinstance(alg, AlgorithmNode)
            execute_algorithm(alg, qc, cbits=cr)
            if is_simulated:
                qc.save_statevector('res', pershot=True)
                state_vs = ibm_simulate_circuit(qc, ibm_noise_model, initial_layout)
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
        

def generate_embeddings(config_path):
    config = load_config_file(config_path, BitflipExperimentID)
    
    if not os.path.exists(config["output_dir"]):
        os.mkdir(config["output_dir"]) 
        
    result = dict()
    c_embeddings = 0
    for hardware_spec in HardwareSpec:
        noise_model = NoiseModel(hardware_spec, thermal_relaxation=WITH_TERMALIZATION)
        if is_hardware_selected(noise_model) and (hardware_spec.value in config["hardware"]):
            assert hardware_spec not in result.keys()
            result[hardware_spec.value] = dict()
            embeddings = get_backend_embeddings(hardware_spec)
            result[hardware_spec.value]["count"] = len(embeddings)
            result[hardware_spec.value]["embeddings"] = embeddings
            c_embeddings += len(embeddings)

    result["count"] = c_embeddings
    f = open(get_embeddings_path(config), "w")
    f.write(json.dumps(result))
    f.close()

def get_embeddings_path(config):
    return os.path.join(config["output_dir"], "embeddings.json")

def load_embeddings(config=None, config_path=None):
    if config is None:
        assert config_path is not None
        config = load_config_file(config_path, BitflipExperimentID)
    
    embeddings_path = get_embeddings_path(config)
    
    experiment_id = config["experiment_id"]
    assert isinstance(experiment_id, BitflipExperimentID)
    
    with open(embeddings_path, 'r') as file:
        result = dict()
        data = json.load(file)
        result["count"] = data["count"]

        for hardware_spec in HardwareSpec:
            if (hardware_spec.value in config["hardware"]):
                noise_model = NoiseModel(hardware_spec, thermal_relaxation=WITH_TERMALIZATION)
                if is_hardware_selected(noise_model):
                
                    result[hardware_spec] = dict()
                    result[hardware_spec]["count"] = data[hardware_spec.value]["count"]
                    result[hardware_spec]["embeddings"] = []

                    for embedding in data[hardware_spec.value]["embeddings"]:
                        d = dict()
                        for (key, value) in embedding.items():
                            d[int(key)] = int(value)
                        if experiment_id == BitflipExperimentID.CXH:
                            temp = d[2]
                            d[2] = d[1]
                            d[1] = temp
                        result[hardware_spec]["embeddings"].append(d)
                else:
                    assert hardware_spec.value not in data.keys()
        
        return result
    raise Exception(f"could not load embeddings file {POMDP_OUTPUT_DIR}{EMBEDDINGS_FILE}")

def get_experiments_actions(noise_model: NoiseModel, embedding: Dict[int,int], experiment_id: BitflipExperimentID):
    if experiment_id == BitflipExperimentID.IPMA:
        if noise_model.basis_gates in [BasisGates.TYPE1]:
            X0 = POMDPAction("X0", [Instruction(embedding[0], Op.U3, params=[pi, 2*pi, pi])])
        else:
            X0 = POMDPAction("X0", [Instruction(embedding[0], Op.X)])
            
        P2 = POMDPAction("P2", [Instruction(embedding[2], Op.MEAS)])
        CX02 = POMDPAction("CX02", [Instruction(embedding[2], Op.CNOT, control=embedding[0])])
        CX12 = POMDPAction("CX12", [Instruction(embedding[2], Op.CNOT, control=embedding[1])])
        return [CX02, CX12, P2, X0]
    else:
        assert experiment_id == BitflipExperimentID.CXH
        if noise_model.basis_gates in [BasisGates.TYPE1, BasisGates.TYPE6]:
            H2 = POMDPAction("H2", [Instruction(embedding[2], Op.U2, params=[0.0, pi])])
            H1 = POMDPAction("H1", [Instruction(embedding[1], Op.U2, params=[0.0, pi])])
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
    

def generate_pomdp(experiment_id: BitflipExperimentID, hardware_spec: HardwareSpec, 
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
    pomdp = build_pomdp(actions, noise_model, 7, embedding, initial_distribution=initial_distribution, guard=guard) # WARNING: 7 is the horizon for which we are interested in this particular experiment for the bitflip problem
    end_time = time.time()
    if return_pomdp:
        return pomdp
    pomdp.serialize(bitflip_instance, pomdp_write_path)
    return end_time-start_time
    
def generate_pomdps(config_path):
    config = load_config_file(config_path, BitflipExperimentID)
    experiment_id = config["experiment_id"]
    assert isinstance(experiment_id, BitflipExperimentID)
    
    # the file that contains the time to generate the POMDP is in this folder
    if not os.path.isdir(config["output_dir"]):
        os.mkdir(config["output_dir"])
        
     # all pomdps will be outputed in this folder:
    output_folder = os.path.join(config["output_dir"], "pomdps")
    # check that there is a folder with the experiment id inside pomdps path
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    all_embeddings = load_embeddings(config=config)
    
    times_file_path = os.path.join(config["output_dir"], 'pomdp_times.csv')
    times_file = open(times_file_path, "w")
    times_file.write("backend,embedding,time\n")
    for backend in HardwareSpec:
        if backend.value in config["hardware"]:
            try:
                embeddings = all_embeddings[backend]["embeddings"]
                for (index, m) in enumerate(embeddings):
                    print(backend, index, m)
                    time_taken = generate_pomdp(experiment_id, backend, m, f"{output_folder}/{backend.value}_{index}.txt")
                    if time_taken is not None:
                        times_file.write(f"{backend.name},{index},{time_taken}\n")
                    times_file.flush()
            except Exception as err:
                print(f"Unexpected {err=}, {type(err)=}")
    times_file.close()

def parse_lambdas_file(config):
    path = os.path.join(config["output_dir"], "lambdas.csv")
    f = open(path)
    result = dict()
    for line in f.readlines()[1:]:
        elements = line.split(",")
        hardware = elements[0]
        embedding_index = int(elements[1])
        horizon = int(elements[2])
        lambda_ = float(elements[3])

        if hardware not in result.keys():
            result[hardware] = dict()
            
        if embedding_index not in result[hardware].keys():
            result[hardware][embedding_index] = dict()
            
        assert horizon not in result[hardware][embedding_index].keys()
        result[hardware][embedding_index][horizon] = lambda_
    f.close()
    return result

def test_programs(config_path, shots=2000, factor=1):
    config = load_config_file(config_path, BitflipExperimentID)
    experiment_id = config["experiment_id"]
    if not os.path.exists(config["output_dir"]):
        raise Exception("output_dir in config does not exists")
    
    lambdas_path = os.path.join(config["output_dir"], 'lambdas.csv')
    if not os.path.exists(lambdas_path):
        raise Exception(f"Guarantees not computed yet (file {lambdas_path} does not exists)")
    
    algorithms_path = os.path.join(config["output_dir"], "algorithms")
    if not os.path.exists(algorithms_path):
        raise Exception(f"Optimal algorithms not computed yet (directory algorithms{experiment_id.value}/ does not exists)")
    
    output_path = os.path.join(config["output_dir"], "real_vs_computed.csv")
    output_file = open(output_path, "w")
    output_file.write("backend,horizon,lambda,acc,diff\n")
    
    all_embeddings = load_embeddings(config=config)
    
    all_lambdas = parse_lambdas_file(config) 
    
    for backend in HardwareSpec: 
        if backend.value in config["hardware"]:
            embeddings = all_embeddings[backend]["embeddings"]
            noise_model = NoiseModel(backend, thermal_relaxation=WITH_TERMALIZATION)
            # we need to be able to translate the actions to the actual instructions for the qpu
            actions_to_instructions = dict()
            actions = get_experiments_actions(noise_model, {0:0, 1:1, 2:2}, experiment_id)
            for action in actions:
                actions_to_instructions[action.name] = action.instruction_sequence
                    
            for (index, embedding) in enumerate(embeddings):
                lambdas_d = all_lambdas[backend.value][index]
                m = embedding
                ibm_bitflip_instance = IBMBitFlipInstance(m)
                
                for horizon in range(config["min_horizon"], config["max_horizon"]+1):
                    algorithm_path = os.path.join(algorithms_path, f"{backend.value}_{index}_{horizon}.json")
                    
                    # load algorithm json
                    f_algorithm = open(algorithm_path)
                    algorithm = AlgorithmNode(serialized=json.load(f_algorithm), actions_to_instructions=actions_to_instructions)
                    f_algorithm.close()  
                            
                    acc = 0
                    for _ in range(factor):
                        acc += ibm_bitflip_instance.ibm_execute_my_algo(algorithm, backend)
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
        all_embeddings = load_embeddings(BitflipExperimentID.IPMA) # FIX ME

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
    def check_instruction_sets(experiment_id: BitflipExperimentID):
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
        
    @staticmethod
    def ibm_run(config_path):
        config = load_config_file(config_path, BitflipExperimentID)
        experiment_id = config["experiment_id"]
        backend = HardwareSpec.TENERIFE
        noise_model = NoiseModel(backend, thermal_relaxation=WITH_TERMALIZATION)
        index = 1
        horizon = 4
        algorithms_path = os.path.join(config["output_dir"], "algorithms")
        algorithm_path = os.path.join(algorithms_path, f"{backend.value}_{index}_{horizon}.json")
        
        # we need to be able to translate the actions to the actual instructions for the qpu
        actions_to_instructions = dict()
        actions = get_experiments_actions(noise_model, {0:0, 1:1, 2:2}, experiment_id)
        for action in actions:
            actions_to_instructions[action.name] = action.instruction_sequence
                
        # load algorithm json
        f_algorithm = open(algorithm_path)
        algorithm = AlgorithmNode(serialized=json.load(f_algorithm), actions_to_instructions=actions_to_instructions)
        f_algorithm.close()
        m = {0: 4, 1: 3, 2: 2}
        ibm_bitflip_instance = IBMBitFlipInstance(m)
        print(ibm_bitflip_instance.ibm_execute_my_algo(algorithm, backend) )


def get_batches():
    s = dict()
    for hardware in HardwareSpec:
        nm = NoiseModel(hardware, thermal_relaxation=WITH_TERMALIZATION)
        if is_hardware_selected(nm):
            if nm.num_qubits not in s.keys():
                s[nm.num_qubits] = []
            s[nm.num_qubits].append(hardware.value) 
    return s
    
def gen_paper_configs():
    # divide experiments into batches
    batches = get_batches()
    
    for (num_qubits, batch) in batches.items():
        # we create a config that will run experiments for all quantum computers that have the same number of qubits
        
        # IPMA
        config_ipma = dict()
        config_ipma["name"] = f"B{num_qubits}"
        config_ipma["experiment_id"] = "ipma"
        config_ipma["min_horizon"] = 4
        config_ipma["max_horizon"] = 7
        config_ipma["output_dir"] = f"/nfs/scistore16/tomgrp/smuroyal/im_time_evolution/synthesis/bitflip/ipma/B{num_qubits}/"
        config_ipma["algorithms_file"] = ""
        config_ipma["hardware"] = batch
        
        ipma_file = open(f"../configs/ipma_b{num_qubits}.json", "w")
        json.dump(config_ipma, ipma_file, indent=4)
        ipma_file.close()
    
        # CX+H
        config_cxh = dict()
        config_cxh["name"] = f"B{num_qubits}"
        config_cxh["experiment_id"] = "cxh"
        config_cxh["min_horizon"] = 4
        config_cxh["max_horizon"] = 7
        config_cxh["output_dir"] = f"/nfs/scistore16/tomgrp/smuroyal/im_time_evolution/synthesis/bitflip/cxh/B{num_qubits}/"
        config_cxh["algorithms_file"] = ""
        config_cxh["hardware"] = batch
        cxh_file = open(f"../configs/cxh_b{num_qubits}.json", "w")
        json.dump(config_cxh, cxh_file, indent=4)
        cxh_file.close()
    
def generate_server_sbatchs():
    f = open("server_script.sh", "w")
    batches = get_batches()
    for num_qubits in batches.keys():
        f.write(f"sbatch sscript.sh {num_qubits}\n")
    f.close()
    
def generate_server_synthesis_script():
    f_ipma = open("../algorithm_synthesis/qalgorithm_synthesis/ipma_script.sh", "w")
    f_cxh = open("../algorithm_synthesis/qalgorithm_synthesis/cxh.sh", "w")
    batches = get_batches()
    for num_qubits in batches.keys():
        f_ipma.write(f"sbatch experiments_script.sh ./input/ipma_b{num_qubits}.input\n")
        f_cxh.write(f"sbatch experiments_script.sh ./input/cxh_b{num_qubits}.input\n")
        
            
    f_ipma.close()
    f_cxh.close()
    
def generate_input_files_for_script():
    batches = get_batches()
    for num_qubits in batches.keys():
        f_ipma = open(f"../algorithm_synthesis/qalgorithm_synthesis/inputs/ipma_b{num_qubits}.input", "w")
        f_cxh = open(f"../algorithm_synthesis/qalgorithm_synthesis/inputs/cxh_b{num_qubits}.input", "w")
        
        f_ipma.write(f"/nfs/scistore16/tomgrp/smuroyal/im_time_evolution/configs/ipma_b{num_qubits}.json\n")
        f_cxh.write(f"/nfs/scistore16/tomgrp/smuroyal/im_time_evolution/configs/cxh_b{num_qubits}.json\n")
        f_ipma.close()
        f_cxh.close()
    
if __name__ == "__main__":
    arg_backend = sys.argv[1]
    Precision.PRECISION = MAX_PRECISION
    Precision.update_threshold()
    if arg_backend == "gen_paper_configs":
        # step 0
        gen_paper_configs()
    elif arg_backend == "embeddings":
        # step 1
        # config_path = sys.argv[2]
        # generate_embeddings(config_path)   

        # generate paper embeddings
        batches = get_batches()
        
        for num_qubits in batches.keys():
            generate_embeddings(f"../configs/ipma_b{num_qubits}.json")
            generate_embeddings(f"../configs/cxh_b{num_qubits}.json")
    elif arg_backend == "all_pomdps":
        # TODO: clean me up
        # step 2: generate all pomdps
        config_path = sys.argv[2]
        # generate_pomdps(config_path)
        
        # generate paper embeddings
        # batches = get_batches()
        
        # for num_qubits in batches.keys():
            # generate_pomdps(f"../configs/ipma_b{num_qubits}.json")
        generate_pomdps(f"../configs/cxh_b{config_path}.json")
        
    # step 3 synthesis of algorithms with C++ code and generate lambdas (guarantees)
    
    elif arg_backend == "simulator_test":
        # step 4: simulate algorithms and compare accuracy with guarantees. Show that it is accurate
        config_path = sys.argv[2]
        test_programs(config_path)
    
    elif arg_backend == "backends_vs":
        # simulate all synthesized algorithms in all backends
        experiment_id = sys.argv[2]


    elif arg_backend == "test" :
        pass     
        # generate_pomdp(BitflipExperimentID.CXH, HardwareSpec.ATHENS, {0: 0, 1: 1, 2: 2}, "", return_pomdp=True)
        # Test.check_selected_hardware()
        # Test.check_embeddings()
        # Test.check_embeddings_file()
        # Test.check_bell_state_creation()
        # Test.check_instruction_sets(BitflipExperimentID.IPMA)
        # Test.check_instruction_sets(BitflipExperimentID.CXH)
        # Test.test_pomdp(HardwareSpec.BOGOTA, 0, BitflipExperimentID.IPMA)
        # Test.test_pomdp(HardwareSpec.TENERIFE, 1, BitflipExperimentID.IPMA)
        # Test.dump_actions(HardwareSpec.TENERIFE, 0, BitflipExperimentID.IPMA)
        # Test.print_pomdp(HardwareSpec.TENERIFE, 0, BitflipExperimentID.IPMA)
        # Test.compare_lambdas(HardwareSpec.TENERIFE)
        # config_path = sys.argv[2]
        # Test.ibm_run(config_path)
        # generate_server_sbatchs()
        generate_server_synthesis_script()
        generate_input_files_for_script()
    else:
        raise Exception("argument does not run any procedure in this script")
        