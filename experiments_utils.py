from contextlib import contextmanager
import json
import os
import signal
from typing import Any, Dict

from ibm_noise_models import HardwareSpec, load_config_file


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
        print("{path} does not exists. Creating ...")
        os.mkdir(path) 
        return False
    return True

def generate_embeddings(**kwargs) -> Dict[Any, Any]:
    config = load_config_file(kwargs["config_path"], kwargs["experiment_enum"])
        
    directory_exists(config["output_dir"])
        
    result = dict()
    c_embeddings = 0
    for hardware_spec in HardwareSpec:
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
        
    