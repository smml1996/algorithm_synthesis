from contextlib import contextmanager
from enum import Enum
import json
from math import ceil
import os
import signal
import time
from typing import Any, Callable, Dict, List

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from algorithm import AlgorithmNode, execute_algorithm
from ibm_noise_models import HardwareSpec, Instruction, NoiseModel, get_ibm_noise_model, ibm_simulate_circuit, load_config_file
from pomdp import POMDPAction, POMDPVertex, build_pomdp, default_guard
import qmemory
from qpu_utils import Op
from utils import find_enum_object, is_matrix_in_list
import subprocess

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
    
########## experiments ids #############
class BitflipExperimentID(Enum):
    IPMA = "ipma"
    IPMA2 = "ipma2"
    CXH = "cxh"
    
    @property
    def exp_name(self):
        return "bitflip"
    
class PhaseflipExperimentID(Enum):
    IPMA = "ipma"
    CXH = "cxh"
    @property
    def exp_name(self):
        return "phaseflip"

class GHZExperimentID(Enum):
    EXP1 = "exp1"
    @property
    def exp_name(self):
        return "ghz"

####### configs ##########
def generate_configs(experiment_id: Enum, min_horizon, max_horizon, allowed_hardware=HardwareSpec, batches: Dict[str, List[HardwareSpec]]=None, opt_technique: str="max", reps=0, verbose=0):
    """_summary_

    Args:
        experiment_id (Enum): _description_
        min_horizon (_type_): _description_
        max_horizon (_type_): _description_
        allowed_hardware (_type_, optional): _description_. Defaults to HardwareSpec.
        batches (Dict[str, List[HardwareSpec]], optional): mapping between batch name and a list of hardware specification for which we perform experiments
        opt_technique (str, optional): possible values are "max" or "min"
        verebose (int, optional): possible values are 1 or 0, meaning verbose equal true and false respectively.
    """    
    experiment_name = experiment_id.exp_name
    assert (min_horizon <= max_horizon)
    configs_path = get_configs_path()
    if not os.path.exists(configs_path):
        print(f"{configs_path} does not exists. Creating it...")
        os.mkdir(configs_path) 
    
    experiment_path = os.path.join(configs_path, f"{experiment_name}")
    if not os.path.exists(experiment_path):
        print(f"{experiment_path} does not exists. Creating it...")
        os.mkdir(experiment_path)
        
    if batches is None:
        batches = get_num_qubits_to_hardware(hardware_str=True, allowed_hardware=allowed_hardware)
    
    for (batch_name, hardware_specs_str) in batches.items():
        if len(hardware_specs_str) > 0:
            config = dict()
            config["name"] = batch_name
            config["experiment_id"] = f"{experiment_id.value}"
            config["min_horizon"] = min_horizon
            config["max_horizon"] = max_horizon
            config["output_dir"] = get_output_path(experiment_name, experiment_id, batch_name)
            config["algorithms_file"] = ""
            config["hardware"] = hardware_specs_str
            config["opt_technique"] = opt_technique
            config["verbose"] = verbose
            config["reps"] = reps
        
            config_path = get_config_path(experiment_id, batch_name)
            f = open(config_path, "w")
            json.dump(config, f, indent=4)
            f.close()

###### embeddings #########

def generate_embeddings(experiment_id, batch, get_hardware_embeddings) -> Dict[Any, Any]:
    config_path = get_config_path(experiment_id, batch)
    config = load_config_file(config_path, type(experiment_id))
    output_dir = os.path.join(get_project_path(), config["output_dir"])
    directory_exists(output_dir)
        
    result = dict()
    c_embeddings = 0
    for hardware_spec_str in config["hardware"]:
        hardware_spec = find_enum_object(hardware_spec_str, HardwareSpec)
        assert hardware_spec not in result.keys()
        result[hardware_spec.value] = dict()
        embeddings = get_hardware_embeddings(hardware_spec, experiment_id)
        result[hardware_spec.value]["count"] = len(embeddings)
        result[hardware_spec.value]["embeddings"] = embeddings
        c_embeddings += len(embeddings)
    result["count"] = c_embeddings
    f = open(get_embeddings_path(config), "w")
    f.write(json.dumps(result))
    f.close()
    
def load_embeddings(config=None, config_path=None):
    if config is None:
        assert config_path is not None
        config = load_config_file(config_path, GHZExperimentID)
    
    embeddings_path = get_embeddings_path(config)
    experiment_id = config["experiment_id"]
    assert isinstance(experiment_id, GHZExperimentID)
    
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
                    result[hardware_spec]["embeddings"].append(d)
            else:
                assert hardware_spec.value not in data.keys()
        return result
    raise Exception(f"could not load embeddings file {embedding_path}")
    
def get_embedding_index(hardware_spec, embedding, experiment_id, get_hardware_embeddings):
    embeddings = get_hardware_embeddings(hardware_spec, experiment_id=experiment_id)
    for (index, embedding_) in enumerate(embeddings):
        if embedding_ == embedding:
            return index
    raise Exception("Embedding not found")

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

def get_num_qubits_to_hardware(hardware_str=True, allowed_hardware=HardwareSpec) -> Dict[int, HardwareSpec|str]:
   
    s = dict()
    for hardware in allowed_hardware:
        nm = NoiseModel(hardware, thermal_relaxation=False)
        if f"B{nm.num_qubits}" not in s.keys():
            s[f"B{nm.num_qubits}"] = []
        if hardware_str:
            s[f"B{nm.num_qubits}"].append(hardware.value) 
        else:
            s[f"B{nm.num_qubits}"].append(hardware) 
    return s

############# about paths ################
def get_project_settings():
    path = "../.settings"
    if not os.path.isfile(path):
        raise Exception("settings file does not exists (are you in /problems directory?)")

    f = open(path, "r")
    answer = dict()
    
    for line in f.readlines():
        elements = line.split(" ")
        assert len(elements) == 3
        answer[elements[0]] = elements[2].rstrip()
        if elements[2] == "None":
            raise Exception("please set properly a project path in .settings file.")
    return answer

def directory_exists(path):
    if not os.path.isdir(path):
        print(f"{path} does not exists. Creating ...")
        os.mkdir(path) 
        return False
    return True

def get_project_path():
    project_settings = get_project_settings()
    return project_settings["PROJECT_PATH"]

def get_configs_path():
    project_path = get_project_path()
    return os.path.join(project_path, "configs")

def get_config_path(experiment_id, batch_name):
    experiment_name = experiment_id.exp_name
    configs_path = get_configs_path()
    experiment_path = os.path.join(configs_path, f"{experiment_name}")
    return os.path.join(experiment_path, f"{experiment_id.value}_{batch_name}.json")

def get_output_path(experiment_name, experiment_id, batch_name):
    project_settings = get_project_settings()
    project_path = project_settings["PROJECT_PATH"]
    directory_exists(os.path.join(project_path, "results"))
    directory_exists(os.path.join(project_path, "results", experiment_name))
    directory_exists(os.path.join(project_path, "results", experiment_name, experiment_id.value))
    return os.path.join("results", experiment_name, experiment_id.value,f"{batch_name}")

def get_pomdp_path(config, hardware_spec, embedding_index):
    return os.path.join(config["output_dir"], "pomdps", f"{hardware_spec.value}_{embedding_index}.txt")

def get_embeddings_path(config):
    return os.path.join(get_project_path(), config["output_dir"], "embeddings.json")


################# C++ code ###############
def get_bellman_value(project_settings, config_path) -> float:
    # # Path to your executable and optional arguments
    executable_path = project_settings["CPP_EXEC_PATH"]
    args = ["bellmaneq", config_path]  # Optional arguments for the executable

    # # Running the executable
    result = subprocess.run([executable_path] + args, capture_output=True, text=True)
    try:
        return float(result.stdout)
    except:
        print(result.stderr)
        raise Exception(f"Could not convert executable to float when running {args}")
    
def get_markov_chain_results(project_settings, algorithm_path, pomdp_path) ->float:
    executable_path = project_settings["CPP_EXEC_PATH"]
    
    args = ["exact", algorithm_path, pomdp_path]
    
    result = subprocess.run([executable_path] + args, capture_output=True, text=True)
    try:
        return float(result.stdout)
    except:
        print("tried mc with args", args)
        print(result.stderr)
        raise Exception(f"Could not convert executable to float when running {args}")

#### default algorithms #####   
def get_default_ghz(noise_model, embedding) -> AlgorithmNode:
    assert isinstance(noise_model, NoiseModel)
    cx01_instruction = Instruction(embedding[1], Op.CNOT, control=embedding[0])
    cx12_instruction = Instruction(embedding[2], Op.CNOT, control=embedding[1])
    if (cx01_instruction in noise_model.instructions_to_channel.keys()) and (cx12_instruction in noise_model.instructions_to_channel.keys()) :
    
        node1 = AlgorithmNode("H0", [Instruction(embedding[0], Op.H)])
        node2 = AlgorithmNode("CX01", [cx01_instruction])
        node3 = AlgorithmNode("CX12", [cx12_instruction])
        node1.next_ins = node2
        node2.next_ins = node3    
        return node1
    else:
        return None

def get_meas_sequence(num_meas, meas_action, flip_action, total_meas, count_ones=0):
    if ceil(total_meas/2.0) <= count_ones:
        return AlgorithmNode(flip_action.name, flip_action.instruction_sequence)
    if num_meas == 0:
        return None
    
    head = AlgorithmNode(meas_action.name, meas_action.instruction_sequence)
    head.case0 = get_meas_sequence(num_meas-1, meas_action, flip_action, total_meas, count_ones)
    head.case1 = get_meas_sequence(num_meas-1, meas_action, flip_action, total_meas, count_ones+1)
    return head
   
def get_default_flip_algorithm(noise_model, embedding, horizon, experiment_id, get_experiments_actions) -> AlgorithmNode:
    if isinstance(experiment_id, BitflipExperimentID):
        experiment_actions = get_experiments_actions(noise_model, embedding, experiment_id)
        flip_action = experiment_actions[2]
        meas_action = experiment_actions[1]
        if experiment_id == BitflipExperimentID.IPMA2:
            head = AlgorithmNode(experiment_actions[0].name, experiment_actions[0].instruction_sequence)
            num_meas = horizon - 2 # the initial CX and the flip instruction needed at the end in case we detect an odd parity
        else:
            raise Exception("Implement me")
            num_meas = horizon - 3 # the two initial CX and the flip instruction needed at the end in case we detect an odd parity
    else:
        assert isinstance(experiment_id, PhaseflipExperimentID)
        experiment_actions = get_experiments_actions(noise_model, embedding, experiment_id)
        flip_action = experiment_actions[1]
        meas_action = experiment_actions[0]
        head = AlgorithmNode(experiment_actions[2].name, experiment_actions[-1].instruction_sequence)
        
        num_meas = horizon-1
    
    
    head.next_ins = get_meas_sequence(num_meas, meas_action, flip_action, total_meas=num_meas)
    return head
        
def get_default_algorithm(noise_model, embedding, experiment_id, get_experiments_actions, horizon):
    if isinstance(experiment_id, GHZExperimentID):
        return get_default_ghz(noise_model, embedding)
    if isinstance(experiment_id, BitflipExperimentID):
        return get_default_flip_algorithm(noise_model, embedding, horizon, BitflipExperimentID.IPMA2, get_experiments_actions)
    return get_default_flip_algorithm(noise_model, embedding, horizon, PhaseflipExperimentID.IPMA, get_experiments_actions)

###### guarantees #####
def get_embedding_guarantee(batch, hardware_spec, embedding_index, horizon, experiment_id):
    ''' This retrieves a guarantee from a file
    '''
    experiment_enum = type(experiment_id)
        
    config_path = get_config_path(experiment_id, batch)
    config = load_config_file(config_path, experiment_enum)
    output_dir = os.path.join(get_project_path(), config["output_dir"])
    lambdas_file_path = os.path.join(output_dir, "lambdas.csv")
    lambdas_file = open(lambdas_file_path)
    
    lines = lambdas_file.readlines()[1:]
    for line in lines:
        elements = line.split(",")
        hardware = elements[0]
        embedding = int(elements[1])
        horizon_ = int(elements[2])
        guarantee = round(float(elements[3]),3)
        if hardware == hardware_spec.value and embedding == embedding_index and horizon == horizon_:
            return guarantee
    return None

def get_custom_guarantee(algorithm_node: AlgorithmNode, pomdp_path, config):
    project_settings = get_project_settings()
    project_path = get_project_path()
    algorithm_path = os.path.join(project_path, config["output_dir"], f"temp_algorithm.json")
    algorithm_file = open(algorithm_path, "w")
    algorithm_file.write(json.dumps(algorithm_node.serialize()))
    algorithm_file.close()
    pomdp_path = os.path.join(project_path, pomdp_path)
    print(pomdp_path)
    print(algorithm_path)
    print("********")
    return get_markov_chain_results(project_settings, algorithm_path, pomdp_path)

def get_guarantees(noise_model: NoiseModel, batch: int, hardware_spec: HardwareSpec, embedding: Dict[int, int], experiment_id: Any, horizon: int, get_experiments_actions, get_hardware_embeddings=None, embedding_index=None):
    config = load_config_file(get_config_path(experiment_id, batch), type(experiment_id))
    if embedding_index is None:
        embedding_index = get_embedding_index(hardware_spec, embedding, experiment_id, get_hardware_embeddings)
    my_guarantee = round(get_embedding_guarantee(batch, hardware_spec, embedding_index, horizon, experiment_id),3)
    
    default_algorithm = get_default_algorithm(noise_model, embedding, experiment_id, get_experiments_actions, horizon)
    if default_algorithm is None:
        # default_algorithm = my_algorithm
        default_guarantee = -1
    else:
        pomdp_path = get_pomdp_path(config, hardware_spec, embedding_index)
        default_guarantee = round(get_custom_guarantee(default_algorithm, pomdp_path, config),3)
    return my_guarantee, default_guarantee

def generate_mc_guarantees_file(experiment_id, allowed_hardware: List[HardwareSpec], get_hardware_embeddings, get_experiments_actions, WITH_THERMALIZATION=False):
    columns = [
        "hardware_spec",
        "embedding_index",
        "horizon",
        "my_guarantee",
        "baseline_guarantee",
        "diff"
    ]
    
    project_path = get_project_path()
    outputfile_path = os.path.join(project_path, "results", experiment_id.exp_name, "mc_guarantees.csv")
    outputfile = open(outputfile_path, "w")
    outputfile.write(",".join(columns) + "\n")
    batches = get_num_qubits_to_hardware(WITH_THERMALIZATION, allowed_hardware=allowed_hardware)
    
    for (batch, hardware_specs) in batches.items():
        config_path = get_config_path(experiment_id, batch)
        config = load_config_file(config_path, type(experiment_id))
        for horizon in range(config["min_horizon"], config["max_horizon"]+1):
            for hardware_spec in hardware_specs:
                noise_model = NoiseModel(hardware_spec, thermal_relaxation=WITH_THERMALIZATION)
                embeddings = get_hardware_embeddings(hardware_spec, experiment_id)
                for (embedding_index, embedding) in enumerate(embeddings):
                    my_guarantee, baseline_guarantee = get_guarantees(noise_model, batch, hardware_spec, embedding, experiment_id, horizon, get_experiments_actions, embedding_index=embedding_index)
                    columns = [hardware_spec.value, embedding_index, horizon, my_guarantee, baseline_guarantee, my_guarantee - baseline_guarantee]
                    columns = [str(x) for x in columns]
                    outputfile.write(",".join(columns) + "\n")
                    
    outputfile.close()
                    
                    

def parse_lambdas_file(config):
    path = os.path.join(get_project_path(), config["output_dir"], "lambdas.csv")
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

### POMDPS ####
def generate_pomdp(experiment_id: GHZExperimentID, hardware_spec: HardwareSpec, 
                embedding: Dict[int, int], pomdp_write_path: str, get_experiments_actions, ProblemInstanceObj, horizon, guard=default_guard,
                return_pomdp=False, 
                WITH_THERMALIZATION=False,
                optimize_graph=True):
    noise_model = NoiseModel(hardware_spec, thermal_relaxation=WITH_THERMALIZATION)
    problem_instance = ProblemInstanceObj(embedding)
    actions = get_experiments_actions(noise_model, embedding, experiment_id)
    initial_distribution = []
    assert len(problem_instance.initial_state) == 1
    initial_distribution.append((problem_instance.initial_state[0], 1.0))
    start_time = time.time()
    pomdp = build_pomdp(actions, noise_model, horizon, embedding, initial_distribution=initial_distribution, guard=guard)
    if optimize_graph:
        pomdp.optimize_graph(problem_instance)
    end_time = time.time()
    if return_pomdp:
        return pomdp
    pomdp.serialize(problem_instance, pomdp_write_path)
    return end_time-start_time

def generate_pomdps(experiment_id, batch, get_experiments_actions, ProblemInstanceObj, guard=default_guard, WITH_THERMALIZATION=False, optimize_graph=True):
    config_path = get_config_path(experiment_id, batch)
    config = load_config_file(config_path, type(experiment_id))
    
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
            embeddings = all_embeddings[backend]["embeddings"]
            
            for (index, m) in enumerate(embeddings):
                print(backend, index, m)
                time_taken = generate_pomdp(experiment_id, backend, m, f"{output_folder}/{backend.value}_{index}.txt", get_experiments_actions, ProblemInstanceObj, config["max_horizon"], guard=guard, WITH_THERMALIZATION=WITH_THERMALIZATION, optimize_graph=optimize_graph)
                if time_taken is not None:
                    times_file.write(f"{backend.name},{index},{time_taken}\n")
                times_file.flush()
    times_file.close()
    
##### guards ####
def bitflips_guard(vertex: POMDPVertex, embedding: Dict[int, int], action: POMDPAction):
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


##### simulations ######

def get_ibm_simulated_acc(alg: AlgorithmNode, embedding, backend: HardwareSpec, IBMProblemInstance,init_states=[0], with_thermalization=False):
    accuracy = 0
    ibm_noise_model = get_ibm_noise_model(backend, thermal_relaxation=with_thermalization)
    ibm_problem_instance = IBMProblemInstance(embedding)
    for init_state in init_states:
        qr = QuantumRegister(ibm_problem_instance.num_qubits)
        cr = ClassicalRegister(ibm_problem_instance.num_bits)
        qc = QuantumCircuit(qr, cr)
        initial_layout = dict()
        for (key, val) in ibm_problem_instance.embedding.items():
            initial_layout[qr[key]] = val
        IBMProblemInstance.prepare_initial_state(qc, init_state)
        assert isinstance(alg, AlgorithmNode)
        execute_algorithm(alg, qc, cbits=cr)
        qc.save_statevector('res', pershot=True)
        state_vs = ibm_simulate_circuit(qc, ibm_noise_model, initial_layout)
        for state in state_vs:
            if ibm_problem_instance.is_target_qs(state):
                accuracy += 1
    return round(accuracy/(1024*len(init_states)), 3)

def compare_with_simulated(experiment_id, batch_name, ibm_instance, get_experiments_actions, factor=1, with_thermalization=False):
    config_path = get_config_path(experiment_id, batch_name)
    config = load_config_file(config_path, type(experiment_id))
    output_dir = os.path.join(get_project_path(), config["output_dir"])
    experiment_id = config["experiment_id"]
    if not os.path.exists(output_dir):
        raise Exception("output_dir in config does not exists")
    
    lambdas_path = os.path.join(output_dir, 'lambdas.csv')
    if not os.path.exists(lambdas_path):
        raise Exception(f"Guarantees not computed yet (file {lambdas_path} does not exists)")
    
    algorithms_path = os.path.join(output_dir, "algorithms")
    if not os.path.exists(algorithms_path):
        raise Exception(f"Optimal algorithms not computed yet (directory algorithms{experiment_id.value}/ does not exists)")
    
    output_path = os.path.join(output_dir, "real_vs_computed.csv")
    output_file = open(output_path, "w")
    output_file.write("backend,horizon,lambda,acc,diff\n")
    
    all_embeddings = load_embeddings(config=config)
    
    all_lambdas = parse_lambdas_file(config) 
    
    for backend in HardwareSpec: 
        if backend.value in config["hardware"]:
            embeddings = all_embeddings[backend]["embeddings"]
            noise_model = NoiseModel(backend, thermal_relaxation=with_thermalization)
            # we need to be able to translate the actions to the actual instructions for the qpu
            actions_to_instructions = dict()
            actions = get_experiments_actions(noise_model, {0:0, 1:1, 2:2}, experiment_id)
            for action in actions:
                actions_to_instructions[action.name] = action.instruction_sequence
            actions_to_instructions["halt"] = []
            for (index, embedding) in enumerate(embeddings):
                lambdas_d = all_lambdas[backend.value][index]
                m = embedding
                ibm_bitflip_instance = ibm_instance(m)
                
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