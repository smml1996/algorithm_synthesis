from contextlib import contextmanager
from enum import Enum
import json
from math import ceil
import os
import signal
import time
from typing import Any, Callable, Dict, List

from algorithm import AlgorithmNode
from ibm_noise_models import HardwareSpec, Instruction, NoiseModel, load_config_file
from qpu_utils import Op
from utils import find_enum_object
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
    
def get_pomdp_path(config, hardware_spec, embedding_index):
    return os.path.join(config["output_dir"], "pomdps", f"{hardware_spec.value}_{embedding_index}.txt")

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

def get_custom_guarantee(algorithm_node: AlgorithmNode, pomdp_path, config):
    project_settings = get_project_settings()
    project_path = get_project_path()
    algorithm_path = os.path.join(project_path, config["output_dir"], f"temp_algorithm.json")
    algorithm_file = open(algorithm_path, "w")
    algorithm_file.write(json.dumps(algorithm_node.serialize()))
    algorithm_file.close()
    pomdp_path = os.path.join(project_path, pomdp_path)
    return get_markov_chain_results(project_settings, algorithm_path, pomdp_path)

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
        head = AlgorithmNode(experiment_actions[0].name, experiment_actions[0].instruction_sequence)
        flip_action = experiment_actions[2]
        meas_action = experiment_actions[1]
    else:
        assert isinstance(experiment_id, PhaseflipExperimentID)
        experiment_actions = get_experiments_actions(noise_model, embedding, experiment_id)
        flip_action = experiment_actions[1]
        meas_action = experiment_actions[0]
        head = AlgorithmNode(experiment_actions[2].name, experiment_actions[-1].instruction_sequence)
    num_meas = horizon-2
    
    
    head.next_ins = get_meas_sequence(num_meas, meas_action, flip_action, total_meas=num_meas)
    return head
        
def get_default_algorithm(noise_model, embedding, experiment_id, get_experiments_actions):
    if isinstance(experiment_id, GHZExperimentID):
        return get_default_ghz(noise_model, embedding)
    if isinstance(experiment_id, BitflipExperimentID):
        return get_default_flip_algorithm(noise_model, embedding, 7, BitflipExperimentID.IPMA2, get_experiments_actions)
    return get_default_flip_algorithm(noise_model, embedding, 7, PhaseflipExperimentID.IPMA, get_experiments_actions)
    
def get_embedding_index(hardware_spec, embedding, experiment_id, get_hardware_embeddings):
    embeddings = get_hardware_embeddings(hardware_spec, experiment_id=experiment_id)
    for (index, embedding_) in enumerate(embeddings):
        if embedding_ == embedding:
            return index
    raise Exception("Embedding not found")

def get_guarantee(batch, hardware_spec, embedding_index, horizon, experiment_id):
    exp_name = experiment_id.exp_name
    experiment_enum = type(experiment_id)
        
    config_path = get_config_path(exp_name, experiment_id, batch)
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

def get_embedding_guarantee(batch, hardware_spec, embedding_index, experiment_id, horizon):        
    return get_guarantee(batch, hardware_spec, embedding_index, horizon, experiment_id)

def get_guarantees(noise_model: NoiseModel, batch: int, hardware_spec: HardwareSpec, embedding: Dict[int, int], experiment_id: Any, horizon: int, get_experiments_actions, get_hardware_embeddings):
    exp_name = experiment_id.exp_name
    config = load_config_file(get_config_path(exp_name, experiment_id, batch), type(experiment_id))
    embedding_index = get_embedding_index(hardware_spec, embedding, experiment_id, get_hardware_embeddings)
    my_guarantee = round(get_embedding_guarantee(batch, hardware_spec, embedding_index, experiment_id, horizon),3)
    
    default_algorithm = get_default_algorithm(noise_model, embedding, experiment_id, get_experiments_actions)
    if default_algorithm is None:
        # default_algorithm = my_algorithm
        default_guarantee = -1
    else:
        pomdp_path = get_pomdp_path(config, hardware_spec, embedding_index)
        default_guarantee = round(get_custom_guarantee(default_algorithm, pomdp_path, config),3)
    return my_guarantee, default_guarantee

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

def get_embeddings_path(config):
    return os.path.join(get_project_path(), config["output_dir"], "embeddings.json")

def directory_exists(path):
    if not os.path.isdir(path):
        print(f"{path} does not exists. Creating ...")
        os.mkdir(path) 
        return False
    return True

def generate_embeddings(**kwargs) -> Dict[Any, Any]:
    config = load_config_file(kwargs["config_path"], kwargs["experiment_enum"])
    output_dir = os.path.join(get_project_path(), config["output_dir"])
    directory_exists(output_dir)
        
    result = dict()
    c_embeddings = 0
    for hardware_spec_str in config["hardware"]:
        hardware_spec = find_enum_object(hardware_spec_str, HardwareSpec)
        assert hardware_spec not in result.keys()
        result[hardware_spec.value] = dict()
        embeddings = kwargs["get_hardware_embeddings"](hardware_spec, **kwargs)
        result[hardware_spec.value]["count"] = len(embeddings)
        result[hardware_spec.value]["embeddings"] = embeddings
        c_embeddings += len(embeddings)

    result["count"] = c_embeddings
    f = open(get_embeddings_path(config), "w")
    f.write(json.dumps(result))
    f.close()
    
def generate_embeddings_files(config: Dict[str, Any], hardware_spec, batch_name:str, embeddings) -> Dict[Any, Any]:
    directory_exists(config["output_dir"])
    
    result = dict()
    result[hardware_spec.value] = dict()
    embedding_index = int(batch_name.split("-")[1]) # assume that embedding index is encoded in the batch name
        
    result[hardware_spec.value]["count"] = 1
    result[hardware_spec.value]["embeddings"] = [embeddings[embedding_index]]

    result["count"] = 1
    f = open(get_embeddings_path(config), "w")
    f.write(json.dumps(result))
    f.close()
    
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

def get_configs_path():
    project_path = get_project_path()
    return os.path.join(project_path, "configs")

def get_project_path():
    project_settings = get_project_settings()
    return project_settings["PROJECT_PATH"]


def get_config_path(experiment_name, experiment_id, batch_name):
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

def generate_configs(experiment_name: str, experiment_id: Enum, min_horizon, max_horizon, allowed_hardware=HardwareSpec, batches: Dict[str, List[HardwareSpec]]=None, opt_technique: str="max", reps=0, verbose=0):
    """_summary_

    Args:
        experiment_name (str): _description_
        experiment_id (Enum): _description_
        min_horizon (_type_): _description_
        max_horizon (_type_): _description_
        allowed_hardware (_type_, optional): _description_. Defaults to HardwareSpec.
        batches (Dict[str, List[HardwareSpec]], optional): mapping between batch name and a list of hardware specification for which we perform experiments
        opt_technique (str, optional): possible values are "max" or "min"
        verebose (int, optional): possible values are 1 or 0, meaning verbose equal true and false respectively.
    """    
    
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
        
            config_path = get_config_path(experiment_name, experiment_id, batch_name)
            f = open(config_path, "w")
            json.dump(config, f, indent=4)
            f.close()

def default_load_embeddings(config, ExperimentIDObj: Enum):
    
    if isinstance(config, str):
        config = load_config_file(config, ExperimentIDObj)
        
    
    embeddings_path = get_embeddings_path(config)
    
    experiment_id = config["experiment_id"]
    assert isinstance(experiment_id, ExperimentIDObj)
    
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
    raise Exception(f"could not load embeddings file {embeddings_path}")


# C++ code    
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

    
