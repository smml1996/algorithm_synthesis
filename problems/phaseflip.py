import os, sys
sys.path.append(os.getcwd()+"/..")

from cmath import isclose
from copy import deepcopy
import time
from typing import Any, Dict, List
import json

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from algorithm import AlgorithmNode, execute_algorithm
from cmemory import ClassicalState
from pomdp import POMDP, POMDPAction, POMDPVertex, build_pomdp
import qmemory
from qpu_utils import GateData, Op, BasisGates
from utils import are_matrices_equal, find_enum_object, get_index, is_matrix_in_list, Precision
sys.path.append(os.getcwd()+"/..")

from qstates import QuantumState
from ibm_noise_models import Instruction, MeasChannel, NoiseModel, get_ibm_noise_model, HardwareSpec, ibm_simulate_circuit, load_config_file
import numpy as np
from math import pi   
from enum import Enum
from experiments_utils import ReadoutNoise, default_load_embeddings, directory_exists, generate_configs, generate_embeddings, get_config_path, get_configs_path, get_embeddings_path, get_num_qubits_to_hardware, get_project_path, get_project_settings
import cProfile
import pstats

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

class PhaseflipExperimentID(Enum):
    IPMA = "ipma"
    CXH = "cxh"


class PhaseFlipInstance:
    def __init__(self, embedding):
        """_summary_

        Args:
            num_qubits (int): _description_
            instruction_set (List[Instruction]): _description_
            embedding (Dict[int, int]): a mapping from logical qubits to physical qubits
        """
        self.embedding = embedding
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


    def get_reward(self, hybrid_state) -> float:
        qs , _ = hybrid_state
        current_rho = qs.single_partial_trace(index=self.embedding[2])
        initial_qs, _ = self.initial_state[0] 
        bell0_rho = initial_qs.single_partial_trace(index=self.embedding[2])
        initial_qs,_ = self.initial_state[1]
        bell1_rho = initial_qs.single_partial_trace(index=self.embedding[2])
        assert len(bell0_rho) == 4
        assert len(bell1_rho) == 4
        if are_matrices_equal(current_rho, bell0_rho) or are_matrices_equal(current_rho, bell1_rho):
            return 1.00
        else:
            return 0.00
    

# choosing embeddings
def get_pivot_qubits(noise_model: NoiseModel, experiment_id: PhaseflipExperimentID):
    result = set()
    noises = []
    if noise_model.hardware_spec == HardwareSpec.MELBOURNE:
        noise_model.num_qubits = 14
    for qubit in range(noise_model.num_qubits):
        if experiment_id == PhaseflipExperimentID.CXH:
            if noise_model.get_qubit_indegree(qubit) > 1:
                noise_data = noise_model.instructions_to_channel[Instruction(qubit, Op.MEAS)]
                assert isinstance(noise_data, MeasChannel)
                success0 = noise_data.get_ind_probability(0,0)
                success1 = noise_data.get_ind_probability(1,1)
                noises.append(ReadoutNoise(qubit, success0, success1))
        else:
            assert experiment_id == PhaseflipExperimentID.IPMA
            if noise_model.get_qubit_outdegree(qubit) > 1:
                noise_data = noise_model.instructions_to_channel[Instruction(qubit, Op.MEAS)]
                assert isinstance(noise_data, MeasChannel)
                success0 = noise_data.get_ind_probability(0,0)
                success1 = noise_data.get_ind_probability(1,1)
                noises.append(ReadoutNoise(qubit, success0, success1))

    temp = sorted(noises, key=lambda x : x.success0)
    result.add(temp[0].target)

    temp = sorted(noises, key=lambda x : x.success1)
    result.add(temp[0].target)

    temp = sorted(noises, key=lambda x: x.acc_err) # accumulated error
    result.add(temp[0].target)

    temp = sorted(noises, key=lambda x: x.diff)
    if temp[0].diff != temp[len(temp)-1].diff:
        result.add(temp[0].target)

    temp = sorted(noises, key=lambda x: x.abs_diff)
    if temp[0].abs_diff != temp[len(temp)-1].abs_diff:
        result.add(temp[0].target)
        assert (temp[0].abs_diff < temp[len(temp)-1].abs_diff)
    return result

def get_selected_couplers(noise_model, target):
    couplers = noise_model.get_qubit_couplers(target)
    first_pair = (couplers[0], couplers[1]) # most noisy pair of couplers for this target
    return first_pair

def does_result_contains_d(result, d):
    for d_ in result:
        controls1 = set([d[0], d[1]])
        controls2 = set([d_[0], d_[1]])
        if d_[2] == d[2] and controls1 == controls2:
            return True
    return False

def get_hardware_embeddings(backend: HardwareSpec, **kwargs) -> List[Dict[int, int]]:
    experiment_id = kwargs["experiment_id"]
    result = []
    noise_model = NoiseModel(backend, thermal_relaxation=WITH_TERMALIZATION)
    assert noise_model.num_qubits >= 14
    pivot_qubits = get_pivot_qubits(noise_model, experiment_id)
    for target in pivot_qubits:
        assert(isinstance(target, int))
        p = get_selected_couplers(noise_model, target)
        d_temp = dict()
        d_temp[0] = p[0][0]
        d_temp[1] = p[1][0]
        d_temp[2] = target
        if not does_result_contains_d(result, d_temp):
            result.append(deepcopy(d_temp))
    return result

class IBMPhaseFlipInstance:
    SHOTS = 1024 * 4
    def __init__(self, embedding) -> None:
        self.embedding = embedding
        new_embedding = dict()
        values = sorted(self.embedding.values())
        for (key, value) in self.embedding.items():
            new_embedding[key] = get_index(value, values)
        self.phaseflip_instance = PhaseFlipInstance(new_embedding)

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

        return self.phaseflip_instance.is_target_qs((qs, None))
    
    def ibm_execute_my_algo(self, alg, backend, log=None):
        accuracy = 0
        
        ibm_noise_model = get_ibm_noise_model(backend, thermal_relaxation=WITH_TERMALIZATION)
        
        if log is not None:
            log_file = open(log, "w")

        for bell_state_index in range(4):
            qr = QuantumRegister(3)
            cr = ClassicalRegister(3)
            qc = QuantumCircuit(qr, cr)
            initial_layout = {qr[0]: self.embedding[0], qr[1]:self.embedding[1], qr[2]:self.embedding[2]}
            IBMPhaseFlipInstance.prepare_bell_state(qc, bell_state_index)
            assert isinstance(alg, AlgorithmNode)
            execute_algorithm(alg, qc, cbits=cr)
            qc.save_statevector('res', pershot=True)
            state_vs = ibm_simulate_circuit(qc, ibm_noise_model, initial_layout)
            for state in state_vs:
                if self.is_target_qs(state):
                    accuracy += 1
        if log is not None:
            log_file.close()
        return round(accuracy/(1024*4), 3)

def load_embeddings(config=None, config_path=None):
    if config is None:
        assert config_path is not None
        config = load_config_file(config_path, PhaseflipExperimentID)
    
    embeddings_path = get_embeddings_path(config)
    
    experiment_id = config["experiment_id"]
    assert isinstance(experiment_id, PhaseflipExperimentID)
    
    with open(embeddings_path, 'r') as file:
        result = dict()
        data = json.load(file)
        result["count"] = data["count"]

        for hardware_spec in HardwareSpec:
            if (hardware_spec.value in config["hardware"]):
                result[hardware_spec] = dict()
                result[hardware_spec]["count"] = data[hardware_spec.value]["count"]
                result[hardware_spec]["embeddings"] = []

                for embedding in data[hardware_spec.value]["embeddings"]:
                    d = dict()
                    for (key, value) in embedding.items():
                        d[int(key)] = int(value)
                    if experiment_id == PhaseflipExperimentID.CXH:
                        temp = d[2]
                        d[2] = d[1]
                        d[1] = temp
                    result[hardware_spec]["embeddings"].append(d)
            else:
                assert hardware_spec.value not in data.keys()
        
        return result
    raise Exception(f"could not load embeddings file {POMDP_OUTPUT_DIR}{EMBEDDINGS_FILE}")


def get_experiments_actions(noise_model: NoiseModel, embedding: Dict[int,int], experiment_id: PhaseflipExperimentID):
    if experiment_id == PhaseflipExperimentID.IPMA:
        if noise_model.basis_gates in [BasisGates.TYPE1, BasisGates.TYPE6]:
            Z0 = POMDPAction("Z0", [Instruction(embedding[0], Op.U3, params=[0.0, 0.0, pi])])
        else:
            Z0 = POMDPAction("Z0", [Instruction(embedding[0], Op.SX), Instruction(embedding[0], Op.RZ, params=[pi]), Instruction(embedding[0], Op.SX)])
            
        if noise_model.basis_gates in [BasisGates.TYPE1, BasisGates.TYPE6]:
            H = POMDPAction("H", [
                Instruction(embedding[2], Op.U2, params=[0.0, pi]),
                Instruction(embedding[1], Op.CNOT, control=embedding[2]),
                Instruction(embedding[0], Op.CNOT, control=embedding[2]),
                Instruction(embedding[2], Op.U2, params=[0.0, pi])
                ])
        else:
            assert noise_model.basis_gates in [BasisGates.TYPE2, BasisGates.TYPE3, BasisGates.TYPE4, BasisGates.TYPE7]
            H = POMDPAction("H", [
                Instruction(embedding[2], Op.RZ, params=[pi/2]),
                Instruction(embedding[2], Op.SX),
                Instruction(embedding[2], Op.RZ, params=[pi/2]),
                Instruction(embedding[1], Op.CNOT, control=embedding[2]),
                Instruction(embedding[0], Op.CNOT, control=embedding[2]),
                Instruction(embedding[2], Op.RZ, params=[pi/2]),
                Instruction(embedding[2], Op.SX),
                Instruction(embedding[2], Op.RZ, params=[pi/2])
            ])
            
        P2 = POMDPAction("P2", [Instruction(embedding[2], Op.MEAS)])
        
        return [P2, Z0, H]
    else:
        assert experiment_id == PhaseflipExperimentID.CXH
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

def generate_pomdp(experiment_id: PhaseflipExperimentID, hardware_spec: HardwareSpec, 
                embedding: Dict[int, int], pomdp_write_path: str, return_pomdp=False):
    noise_model = NoiseModel(hardware_spec, thermal_relaxation=WITH_TERMALIZATION)
    phaseflip_instance = PhaseFlipInstance(embedding)
    actions = get_experiments_actions(noise_model, embedding, experiment_id)
    initial_distribution = []
    for s in phaseflip_instance.initial_state:
        initial_distribution.append((s, 0.25))

    start_time = time.time()
    pomdp = build_pomdp(actions, noise_model, 7, embedding, initial_distribution=initial_distribution, guard=guard) # WARNING: 7 is the horizon for which we are interested in this particular experiment for the phaseflip problem
    pomdp.optimize_graph(phaseflip_instance)
    end_time = time.time()
    if return_pomdp:
        return pomdp
    pomdp.serialize(phaseflip_instance, pomdp_write_path)
    return end_time-start_time
    
def generate_pomdps(config_path):
    config = load_config_file(config_path, PhaseflipExperimentID)
    experiment_id = config["experiment_id"]
    assert isinstance(experiment_id, PhaseflipExperimentID)
    
    # the file that contains the time to generate the POMDP is in this folder
    output_dir = os.path.join(get_project_path(), config["output_dir"])
    directory_exists(output_dir)
        
     # all pomdps will be outputed in this folder:
    output_folder = os.path.join(output_dir, "pomdps")
    # check that there is a folder with the experiment id inside pomdps path
    directory_exists(output_folder)

    all_embeddings = load_embeddings(config=config)
    
    times_file_path = os.path.join(output_dir, 'pomdp_times.csv')
    times_file = open(times_file_path, "w")
    times_file.write("backend,embedding,time\n")
    for backend in HardwareSpec:
        if backend.value in config["hardware"]:
            # try:
            embeddings = all_embeddings[backend]["embeddings"]
            for (index, m) in enumerate(embeddings):
                print(backend, index, m)
                time_taken = generate_pomdp(experiment_id, backend, m, f"{output_folder}/{backend.value}_{index}.txt")
                if time_taken is not None:
                    times_file.write(f"{backend.name},{index},{time_taken}\n")
                times_file.flush()
            # except Exception as err:
            #     print(f"Unexpected {err=}, {type(err)=}")
    times_file.close()

def test_programs(config_path, shots=2000, factor=1):
    config = load_config_file(config_path, PhaseflipExperimentID)
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
            actions_to_instructions["halt"] = []
            for (index, embedding) in enumerate(embeddings):
                lambdas_d = all_lambdas[backend.value][index]
                m = embedding
                ibm_phaseflip_instance = IBMPhaseFlipInstance(m)
                
                for horizon in range(config["min_horizon"], config["max_horizon"]+1):
                    algorithm_path = os.path.join(algorithms_path, f"{backend.value}_{index}_{horizon}.json")
                    
                    # load algorithm json
                    f_algorithm = open(algorithm_path)
                    algorithm = AlgorithmNode(serialized=json.load(f_algorithm), actions_to_instructions=actions_to_instructions)
                    f_algorithm.close()  
                            
                    acc = 0
                    for _ in range(factor):
                        acc += ibm_phaseflip_instance.ibm_execute_my_algo(algorithm, backend)
                    acc /= factor  
                    acc = round(acc, 3)
                    output_file.write(f"{backend}-{index},{horizon},{lambdas_d[horizon]},{acc},{round(lambdas_d[horizon]-acc,3)}\n")
                    output_file.flush()
    output_file.close()
    
    
if __name__ == "__main__":
    arg_backend = sys.argv[1]
    Precision.PRECISION = MAX_PRECISION
    Precision.update_threshold()
    settings = get_project_settings()
    project_path = settings["PROJECT_PATH"]
    
    allowed_hardware = []
    for hardware in HardwareSpec:
        noise_model = NoiseModel(hardware, thermal_relaxation=WITH_TERMALIZATION)
        if noise_model.num_qubits >= 14:
            allowed_hardware.append(hardware)
    if arg_backend == "gen_configs":
        # step 0
        generate_configs(experiment_name="phaseflip", experiment_id=PhaseflipExperimentID.IPMA, min_horizon=4, max_horizon=7, allowed_hardware=allowed_hardware)
        generate_configs(experiment_name="phaseflip", experiment_id=PhaseflipExperimentID.CXH, min_horizon=4, max_horizon=7, allowed_hardware=allowed_hardware)
    elif arg_backend == "embeddings":
        # generate paper embeddings
        batches = get_num_qubits_to_hardware(WITH_TERMALIZATION, allowed_hardware=allowed_hardware)
        for num_qubits in batches.keys():
            ipma_config_path = get_config_path("phaseflip", PhaseflipExperimentID.IPMA, num_qubits)
            generate_embeddings(config_path=ipma_config_path, experiment_enum=PhaseflipExperimentID, get_hardware_embeddings=get_hardware_embeddings, experiment_id=PhaseflipExperimentID.IPMA)
            
            cxh_config_path = get_config_path("phaseflip", PhaseflipExperimentID.CXH, num_qubits)
            generate_embeddings(config_path=cxh_config_path, experiment_enum=PhaseflipExperimentID, get_hardware_embeddings=get_hardware_embeddings, experiment_id=PhaseflipExperimentID.CXH)
    elif arg_backend == "all_pomdps":
        # TODO: clean me up
        # step 2: generate all pomdps
        # config_path = sys.argv[2]
        # generate_pomdps(config_path)
        
        # generate paper embeddings
        batches = get_num_qubits_to_hardware(WITH_TERMALIZATION, allowed_hardware=allowed_hardware)
        for num_qubits in batches.keys():
            generate_pomdps(get_config_path("phaseflip", PhaseflipExperimentID.IPMA, num_qubits))
            # generate_pomdps(get_config_path("phaseflip", PhaseflipExperimentID.CXH, num_qubits))
        
    # step 3 synthesis of algorithms with C++ code and generate lambdas (guarantees)
    
    elif arg_backend == "simulator_test":
        # step 4: simulate algorithms and compare accuracy with guarantees. Show that it is accurate
        config_path = sys.argv[2]
        test_programs(config_path)
    
    elif arg_backend == "backends_vs":
        # simulate all synthesized algorithms in all backends
        experiment_id = sys.argv[2]
    else:
        raise Exception("argument does not run any procedure in this script")
        