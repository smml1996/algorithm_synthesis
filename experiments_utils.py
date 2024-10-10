from contextlib import contextmanager
from enum import Enum
import json
import os
import signal
from typing import Any, Dict

from ibm_noise_models import HardwareSpec, NoiseModel, load_config_file
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
    
def get_num_qubits_to_hardware(hardware_str=True) -> Dict[int, HardwareSpec|str]:
    s = dict()
    for hardware in HardwareSpec:
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

def generate_configs(experiment_name: str, experiment_id: Enum, min_horizon, max_horizon):
    configs_path = get_configs_path()
    if not os.path.exists(configs_path):
        print(f"{configs_path} does not exists. Creating it...")
        os.mkdir(configs_path) 
    
    experiment_path = os.path.join(configs_path, f"{experiment_name}")
    if not os.path.exists(experiment_path):
        print(f"{experiment_path} does not exists. Creating it...")
        os.mkdir(experiment_path)
        
    batches = get_num_qubits_to_hardware(hardware_str=True)
    
    
    for (num_qubits, hardware_specs_str) in batches.items():
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
        
    