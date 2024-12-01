import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from experiments_utils import *

###### POMDPs SIZES AND TIMES ######
class EXPAttr(Enum):
    POMDP_SIZE = "pomdp_size"
    POMDP_TIME = "pomdp_time"
    BELLMAN_TIME = "bellman_time"
    
def get_pomdp_num_states(config, hardware_spec, embedding_index):
    pomdp_path = get_pomdp_path(config, hardware_spec, embedding_index)
    f = open(pomdp_path)
    lines = f.readlines()
    states_line = lines[2]
    elements = states_line.split(" ")
    assert elements[0] == "STATES:"
    f.close()
    return len(elements[1].split(","))

def get_exp_pomdps_attr(experiment_id, attr: EXPAttr, with_thermalization=False):
    answer = []
    batches = get_num_qubits_to_hardware(with_thermalization, allowed_hardware=get_allowed_hardware(experiment_id))
    
    for (batch_name, hardware_specs) in batches.items():
        config_path = get_config_path(experiment_id, batch_name)
        config = load_config_file(config_path, type(experiment_id))
        embeddings = load_embeddings(config)
        for hardware_spec in hardware_specs:
            assert hardware_spec in embeddings.keys()
            for embedding_index in range(len(embeddings[hardware_spec]["embeddings"])):
                if attr == EXPAttr.POMDP_SIZE:
                    answer.append(get_pomdp_num_states(config, hardware_spec, embedding_index))
                else:
                    raise Exception(f"not valid experiment attribute {attr}")
    return answer
            
def get_pomdp_times(experiment_id, with_thermalization=False):
    project_path = get_project_path()
    batches = get_num_qubits_to_hardware(with_thermalization, allowed_hardware=get_allowed_hardware(experiment_id))
    answer = []
    for batch_name in batches.keys():
        output_path = get_output_path(experiment_id, batch_name)
        times_file_path = os.path.join(project_path, output_path, "pomdp_times.csv")
        f = open(times_file_path)
        lines = f.readlines()
        for line in lines[1:]:
            elements = line.split(",")
            answer.append(float(elements[2]))
        f.close()
    return answer

def get_bellman_times(experiment_id, with_thermalization=False):
    project_path = get_project_path()
    batches = get_num_qubits_to_hardware(with_thermalization, allowed_hardware=get_allowed_hardware(experiment_id))
    times = []
    horizons = []
    for batch_name in batches.keys():
        output_path = get_output_path(experiment_id, batch_name)
        times_file_path = os.path.join(project_path, output_path, "lambdas.csv")
        f = open(times_file_path)
        lines = f.readlines()
        for line in lines[1:]:
            elements = line.split(",")
            horizons.append(int(elements[2]))
            times.append(float(elements[4]))
        f.close()
    assert len(horizons) == len(times)
    return horizons, times

def get_all_experiments_attr(experiments_ids, attr: EXPAttr, with_thermalization=False) -> pd.DataFrame:
    counts = []
    experiment_names = []
    horizons = []
    for experiment_id in experiments_ids:
        if attr == EXPAttr.POMDP_SIZE:
            temp = get_exp_pomdps_attr(experiment_id, attr, with_thermalization=with_thermalization)
        elif attr == EXPAttr.POMDP_TIME:
            temp = get_pomdp_times(experiment_id)
        elif attr == EXPAttr.BELLMAN_TIME:
            horizons_, times = get_bellman_times(experiment_id)
        else:
            raise Exception(f"not valid attribute {attr}")
        
        if attr in [EXPAttr.POMDP_SIZE, EXPAttr.POMDP_TIME]:
            for t in temp:
                counts.append(t)
                experiment_names.append(f"{experiment_id.exp_name}-{experiment_id.value}")
        elif attr  == EXPAttr.BELLMAN_TIME:
            for (horizon, time) in zip(horizons_, times):
                experiment_names.append(f"{experiment_id.exp_name}-{experiment_id.value}")
                counts.append(time)
                horizons.append(horizon)
        else:
            raise Exception (f"Invalid attr {attr}")
    if attr in [EXPAttr.POMDP_SIZE, EXPAttr.POMDP_TIME]:
        return pd.DataFrame.from_dict({
            'experiment': experiment_names,
            attr.value: counts,
        }) 
    return  pd.DataFrame.from_dict({
            'experiment': experiment_names,
            'horizon': horizons,
            attr.value: counts,
        }) 
        