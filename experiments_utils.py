from contextlib import contextmanager
from enum import Enum
import json
import os
import signal
import time
from typing import Any, Callable, Dict, List

from ibm_noise_models import HardwareSpec, NoiseModel, load_config_file
from pomdp import POMDPAction, POMDPVertex, build_pomdp
from utils import find_enum_object


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
        raise Exception("settings file does not exists")

    f = open(path, "r")
    answer = dict()
    
    for line in f.readlines():
        elements = line.split(" ")
        assert len(elements) == 3
        answer[elements[0]] = elements[2]
        
    return answer

def get_embeddings_path(config):
    return os.path.join(config["output_dir"], "embeddings.json")

def directory_exists(path):
    if not os.path.isdir(path):
        print(f"{path} does not exists. Creating ...")
        os.mkdir(path) 
        return False
    return True

def generate_embeddings(**kwargs) -> Dict[Any, Any]:
    config = load_config_file(kwargs["config_path"], kwargs["experiment_enum"])
        
    directory_exists(config["output_dir"])
        
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
    
def get_num_qubits_to_hardware(hardware_str=True, allowed_hardware=HardwareSpec) -> Dict[int, HardwareSpec|str]:
    s = dict()
    for hardware in allowed_hardware:
        nm = NoiseModel(hardware, thermal_relaxation=False)
        if nm.num_qubits not in s.keys():
            s[nm.num_qubits] = []
        if hardware_str:
            s[nm.num_qubits].append(hardware.value) 
        else:
            s[nm.num_qubits].append(hardware) 
    return s

def get_configs_path():
    project_settings = get_project_settings()
    project_path = project_settings["PROJECT_PATH"]
    return os.path.join(project_path, "configs")

def get_config_path(experiment_name, experiment_id, batch_number):
    
    configs_path = get_configs_path()
    experiment_path = os.path.join(configs_path, f"{experiment_name}")
    return os.path.join(experiment_path, f"{experiment_id.value}_{batch_number}.json")

def get_output_path(experiment_name, experiment_id, batch_number):
    project_settings = get_project_settings()
    project_path = project_settings["PROJECT_PATH"]
    directory_exists(os.path.join(project_path, "synthesis"))
    directory_exists(os.path.join(project_path, "synthesis", experiment_name))
    directory_exists(os.path.join(project_path, "synthesis", experiment_name, experiment_id.value))
    return os.path.join(project_path, "synthesis", experiment_name, experiment_id.value,f"B{batch_number}")

def generate_configs(experiment_name: str, experiment_id: Enum, min_horizon, max_horizon, allowed_hardware=HardwareSpec):
    configs_path = get_configs_path()
    if not os.path.exists(configs_path):
        print(f"{configs_path} does not exists. Creating it...")
        os.mkdir(configs_path) 
    
    experiment_path = os.path.join(configs_path, f"{experiment_name}")
    if not os.path.exists(experiment_path):
        print(f"{experiment_path} does not exists. Creating it...")
        os.mkdir(experiment_path)
        
    batches = get_num_qubits_to_hardware(hardware_str=True, allowed_hardware=allowed_hardware)
    
    
    for (num_qubits, hardware_specs_str) in batches.items():
        if len(hardware_specs_str) > 0:
            config = dict()
            config["name"] = f"B{num_qubits}"
            config["experiment_id"] = f"{experiment_id.value}"
            config["min_horizon"] = min_horizon
            config["max_horizon"] = max_horizon
            config["output_dir"] = get_output_path(experiment_name, experiment_id, num_qubits)
            config["algorithms_file"] = ""
            config["hardware"] = hardware_specs_str
        
            config_path = get_config_path(experiment_name, experiment_id, num_qubits)
            f = open(config_path, "w")
            json.dump(config, f, indent=4)
            f.close()
            
def generate_pomdp(experiment_id: Any, ProblemInstance,  hardware_spec: HardwareSpec, 
                embedding: Dict[int, int], get_experiments_actions:Callable[[NoiseModel, Dict[int, int], Enum], List[POMDPAction]], 
                guard: Callable[[POMDPVertex, Dict[int, int], POMDPAction], bool], 
                max_horizon: int, thermal_relaxation: bool, 
                pomdp_write_path: str, return_pomdp=False, **kwargs):
    noise_model = NoiseModel(hardware_spec, thermal_relaxation=thermal_relaxation)
    print(kwargs)
    problem_instance = ProblemInstance(kwargs)
    actions = get_experiments_actions(noise_model, embedding, experiment_id)
    initial_distribution = []
    for s in problem_instance.initial_state:
        initial_distribution.append((s, 1/len(problem_instance.initial_state)))

    start_time = time.time()
    pomdp = build_pomdp(actions, noise_model, max_horizon, embedding, initial_distribution=initial_distribution, guard=guard)
    pomdp.optimize_graph(problem_instance)
    end_time = time.time()
    if return_pomdp:
        return pomdp
    pomdp.serialize(problem_instance, pomdp_write_path)
    return end_time-start_time

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


def generate_pomdps(config_path: str, ProblemInstance: Any, ExperimentIdObj: Enum, 
                    get_experiments_actions:Callable[[NoiseModel, Dict[int, int], Enum], List[POMDPAction]], 
                    guard: Callable[[POMDPVertex, Dict[int, int], POMDPAction], bool],
                    load_embeddings: Callable[[str, Enum], List[Dict[int, int]]]=default_load_embeddings, 
                    thermal_relaxation=False, **kwargs):
    config = load_config_file(config_path, ExperimentIdObj)
    experiment_id = config["experiment_id"]
    assert isinstance(experiment_id, ExperimentIdObj)
    
    max_horizon = config["max_horizon"]
    assert isinstance(max_horizon, int)
    
    # the file that contains the time to generate the POMDP is in this folder
    directory_exists(config["output_dir"])
        
     # all pomdps will be outputed in this folder:
    output_folder = os.path.join(config["output_dir"], "pomdps")
    # check that there is a folder with the experiment id inside pomdps path
    directory_exists(output_folder)

    all_embeddings = load_embeddings(config, ExperimentIdObj)
    
    times_file_path = os.path.join(config["output_dir"], 'pomdp_times.csv')
    times_file = open(times_file_path, "w")
    times_file.write("backend,embedding,time\n")
    for backend in HardwareSpec:
        if backend.value in config["hardware"]:
            # try:
            embeddings = all_embeddings[backend]["embeddings"]
            for (index, m) in enumerate(embeddings):
                print(kwargs)
                kwargs['kwargs']["embedding"] = m
                print(backend, index, m)
                print("after",kwargs)
                time_taken = generate_pomdp(experiment_id, ProblemInstance, backend, m, get_experiments_actions, guard, max_horizon, thermal_relaxation, f"{output_folder}/{backend.value}_{index}.txt", kwargs)
                if time_taken is not None:
                    times_file.write(f"{backend.name},{index},{time_taken}\n")
                times_file.flush()
            # except Exception as err:
            #     print(f"Unexpected {err=}, {type(err)=}")
    times_file.close()
        
    