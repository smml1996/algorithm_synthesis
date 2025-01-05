from importlib.machinery import SourceFileLoader
import os, sys
from typing import Tuple
import copy
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from utils import Precision
from experiments_utils import *
from ibm_noise_models import MeasChannel, QuantumChannel
from qpu_utils import is_multiqubit_gate

Precision.PRECISION = 8
Precision.update_threshold()

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
        
### DIFF ALGORITHMS ANALYSIS ###
def get_num_algorithms(path) -> List[AlgorithmNode]:
    mod = SourceFileLoader("m", path).load_module()
    return len(mod.algorithms)

def get_dict_horizon_num_algs(experiment_id, min_horizon, max_horizon) -> Dict[int, int]:
    answer = dict()
    algos_path = get_experiment_name_path(experiment_id)
    project_path = get_project_path()
    for horizon in range(min_horizon, max_horizon+1):
        algo_path = os.path.join(project_path, algos_path, experiment_id.value, f"diff{horizon}_algs.py")
        answer[horizon] = get_num_algorithms(algo_path)
    return answer

def get_max_diffs_dicts(horizon_dict, algorithm_index, num_algorithms):
    answer = dict()
    for alg_index_ in range(num_algorithms):
        accs_diffs = []
        for (backend, b_dict) in horizon_dict.items():
            accs_diffs.append(abs(b_dict[algorithm_index] - b_dict[alg_index_] ))

        assert alg_index_ not in answer.keys()
        answer[alg_index_] = round(max(accs_diffs), 2)
    return answer
        
def parse_algorithms_vs_file(experiment_id):
    answer = dict() # horizon -> hardware -> alg_index -> acc
    project_path = get_project_path()
    experiment_path = get_experiment_name_path(experiment_id)
    algorithms_vs_path = os.path.join(project_path, experiment_path, experiment_id.value, "algorithms_vs.csv")
    algorithms_vs_file = open(algorithms_vs_path)
    lines = algorithms_vs_file.readlines()
    algorithms_vs_file.close()
    for line in lines[1:]:
        elements = line.split(",")
        horizon = int(elements[0])
        alg_index = int(elements[1])
        hardware_spec = elements[2] + "-" + elements[3] # hardware_spec + embedding_index
        acc = float(elements[4])
        if horizon not in answer.keys():
            answer[horizon] = dict()
        if hardware_spec not in answer[horizon].keys():
            answer[horizon][hardware_spec] = dict()
        
        assert alg_index not in answer[horizon][hardware_spec]
        answer[horizon][hardware_spec][alg_index] = acc
    return answer
            
    
def get_horizon_max_diffs(experiment_id, min_horizon, max_horizon):
    horizon_to_num_algs = get_dict_horizon_num_algs(experiment_id, min_horizon, max_horizon)
    
    horizons_dict = parse_algorithms_vs_file(experiment_id)
    
    answer = dict()
    for horizon in range(min_horizon, max_horizon+1):
        assert horizon not in answer.keys()
        answer[horizon] = dict()
        for alg_index in range(0, horizon_to_num_algs[horizon]):
            answer[horizon][alg_index] = get_max_diffs_dicts(horizons_dict[horizon], alg_index, horizon_to_num_algs[horizon])
    return answer

def find_similar_algorithms(experiment_id, min_horizon, max_horizon):
    max_diffs = get_horizon_max_diffs(experiment_id, min_horizon, max_horizon)
    
    for horizon in range(min_horizon, max_horizon+1):
        for (alg_index1, comparison_dict) in max_diffs[horizon].items():
            for (alg_index2, diff) in comparison_dict.items():
                if alg_index1 < alg_index2:
                    if diff < 0.01:
                        print(f"[h={horizon}] {alg_index1} ~ {alg_index2} --> diff. = {diff}")
                        
def get_diff_algorithms_to_spec(experiment_id, get_hardware_embeddings, get_experiments_actions, with_thermalization=False):
    project_path = get_project_path()
    allowed_hardware = get_allowed_hardware(experiment_id, with_thermalization=with_thermalization)
    all_algorithms = dict()
    
    answer = dict() # horizon --> alg_index -->  List[(HardwareSpec, embedding_index)]

    batches = get_num_qubits_to_hardware(with_thermalization, allowed_hardware=allowed_hardware)
    for (batch, hardware_specs) in batches.items():
        config_path = get_config_path(experiment_id, batch)
        config = load_config_file(config_path, type(experiment_id))
        all_algorithms_path = os.path.join(project_path, config["output_dir"], "algorithms")
        min_horizon = config["min_horizon"]
        max_horizon = config["max_horizon"]
        for hardware_spec in hardware_specs:
            embeddings = get_hardware_embeddings(hardware_spec, experiment_id)
            noise_model = NoiseModel(hardware_spec, thermal_relaxation=with_thermalization)
                
            for (index, embedding) in enumerate(embeddings):
                actions_to_instructions = dict()
                actions = get_experiments_actions(noise_model, get_default_embedding(len(embedding.keys())), experiment_id)
                for action in actions:
                    actions_to_instructions[action.name] = action.instruction_sequence
                actions_to_instructions["halt"] = []
                for horizon in range(min_horizon, max_horizon+1):
                    if horizon not in all_algorithms:
                        all_algorithms[horizon] = []
                        assert horizon not in answer.keys()
                        answer[horizon] = dict()
                    
                    algorithm_path = os.path.join(all_algorithms_path, f"{hardware_spec.value}_{index}_{horizon}.json")
                    f_algorithm = open(algorithm_path)
                    algorithm = AlgorithmNode(serialized=json.load(f_algorithm), actions_to_instructions=actions_to_instructions)
                    f_algorithm.close()
                    if not (algorithm in all_algorithms[horizon]):
                        all_algorithms[horizon].append(algorithm)
                        answer[horizon][len(all_algorithms[horizon])-1] = [(hardware_spec, index)]
                    else:
                        for (index_old_alg, old_alg) in enumerate(all_algorithms[horizon]):
                            if old_alg == algorithm:
                                answer[horizon][index_old_alg].append((hardware_spec, index))
                                break
    return answer
    
def get_meas_avgs(meas_instructions: List[Instruction], noise_model: NoiseModel) -> Tuple[float, float, float, float]:
    if len(meas_instructions) == 0:
        return 0, 0, 0, 0
    success0 = 0.0
    success1 = 0.0
    diff = 0.0
    accum = 0.0
    for meas_instruction in meas_instructions:
        noise_data = noise_model.instructions_to_channel[meas_instruction]
        assert isinstance(noise_data, MeasChannel)
        success0 = noise_data.get_ind_probability(0, 0)
        success1 = noise_data.get_ind_probability(1, 1)
        diff = success0 - success1
        accum = (success0 + success1)/2
    
    success0 = success0 / len(meas_instructions)
    success1 = success1 / len(meas_instructions)
    diff = diff / len(meas_instructions)
    accum = accum / len(meas_instructions)
    return success0, success1, diff, accum
    

def get_avgs_unitaries(instructions: List[Instruction], noise_model: NoiseModel) -> float:
    avg = 0.0
    if len(instructions) == 0:
        return 0.0
    
    for instruction in instructions:
        channel = noise_model.instructions_to_channel[instruction]
        assert isinstance(channel, QuantumChannel)
        avg += channel.estimated_success_prob
    return avg / len(instructions)

def get_ops(pomdp_actions: List[POMDPAction]) -> Tuple[List[Instruction], List[Instruction], List[Instruction]]:
    meas_instructions = []
    couplers = []
    oneQ_gates = []
    
    for pomdp_action in pomdp_actions:
        for instruction in pomdp_action.instruction_sequence:
            assert isinstance(instruction, Instruction)
            if instruction.is_meas_instruction():
                meas_instructions.append(instruction)
            elif instruction.control is None:
                oneQ_gates.append(instruction)
            else:
                couplers.append(instruction)
    
    return meas_instructions, couplers, oneQ_gates


def get_hard_algos_analysis_df(experiment_id, get_experiments_actions, horizon, algs_to_hardware_: Dict[int, List[HardwareSpec]]):
    '''algs_to_hardware is a dictionary mapping horizon to alg_index to a list of hardware specification for which that algorithm is optimal. It is obtained from get_diff_algotihsm
    '''
    all_embeddings = load_all_embeddings(experiment_id)
    algs_to_hardware = algs_to_hardware_[horizon]
    backends = []
    horizons = []
    algorithms_index = []
    avgs_p0s = []
    avgs_p1s = []
    avgs_diffs = []
    avgs_probs_accum = []
    avgs_couplers_success_probs = []
    avgs_1q_success = []
    noise_model = None
    
    for (alg_index, algo_backends) in algs_to_hardware.items():
            for (hardware_spec, embedding_index) in algo_backends:
                if (noise_model is None) or (noise_model.hardware_spec != hardware_spec):
                    noise_model = NoiseModel(hardware_spec, thermal_relaxation=False)
                horizons.append(horizon)
                backends.append(hardware_spec.value)
                algorithms_index.append(alg_index)
                embedding = all_embeddings[hardware_spec][embedding_index]
                pomdp_actions = get_experiments_actions(noise_model, embedding, experiment_id)
                
                meas_instructions, couplers, oneQ_gates = get_ops(pomdp_actions)
                success0, success1, diff, accum = get_meas_avgs(meas_instructions, noise_model)
                avgs_p0s.append(round(success0, 3))
                avgs_p1s.append(round(success1, 3))
                avgs_diffs.append(round(diff, 3))
                avgs_probs_accum.append(round(accum, 3))
    
                avg_couplers = get_avgs_unitaries(couplers, noise_model)
                avgs_couplers_success_probs.append(round(avg_couplers,3))
                
                avg_1Q = get_avgs_unitaries(oneQ_gates, noise_model)
                avgs_1q_success.append(avg_1Q)
    
    
    
    df = pd.DataFrame.from_dict({
        'hardware_spec': backends,
        'program': algorithms_index,
        'success0': avgs_p0s,
        'success1': avgs_p1s,
        'success_diff': avgs_diffs,
        'accum_prob': avgs_probs_accum,
        'couplers_success': avgs_couplers_success_probs,
        'oneq_gates': avgs_1q_success
    }) 
    return df

def get_summary_per_algo(experiment_id, get_experiments_actions, horizon, algs_to_hardware_: Dict[int, List[HardwareSpec]], drop_columns = []):
    df = get_hard_algos_analysis_df(experiment_id, get_experiments_actions, horizon, algs_to_hardware_)
    df.drop(drop_columns, axis=1, inplace=True)
    all_columns = ['success0', 'success1', 'success_diff', 'accum_prob', 'couplers_success', 'oneq_gates']
    final_list = []
    for c in all_columns:
        if c not in drop_columns:
            final_list.append(c)
    df_melted = df.melt(id_vars=["hardware_spec", "program"], value_vars=final_list)
    result = df_melted.groupby(['program', 'variable']).describe()
    return result.round(3)

def get_best_algos_advantanges(experiment_id, diff_algs_to_spec, mc_guarantees_file_name="mc_guarantees_.csv", take_best=True):
    hard_spec_to_alg_index = get_hard_spec_to_alg_index(diff_algs_to_spec)
    
    # load mc_guarantees file
    file_path = os.path.join(get_project_path(), get_experiment_name_path(experiment_id), experiment_id.value, mc_guarantees_file_name)
    f = open(file_path)
    lines = f.readlines()[1:]
    f.close()
    
    answer = dict()
    for line in lines:
        elements = line.split(",")
        hardware = elements[0]
        embedding_index = int(elements[1])
        horizon = int(elements[2])
        my_guarantee = float(elements[3])
        trad_guarantee = float(elements[4])
        best_trad_guarantee = float(elements[6])
        alg_index = hard_spec_to_alg_index[horizon][find_enum_object(hardware, HardwareSpec)][embedding_index]
        if horizon not in answer.keys():
            answer[horizon] = dict()
            
        if take_best:
            final_trad_guarantee = best_trad_guarantee
        else:
            final_trad_guarantee = trad_guarantee
            
        assert my_guarantee >= final_trad_guarantee
        
        if my_guarantee > final_trad_guarantee:
            if alg_index not in answer[horizon].keys():
                answer[horizon][alg_index] = (my_guarantee - final_trad_guarantee)
            else:
                answer[horizon][alg_index] = max((my_guarantee - final_trad_guarantee), answer[horizon][alg_index])
    return answer

### GRAPHS ###
def get_hard_spec_to_alg_index(diff_algs_to_spec):
    # diff_algs_to_spec is a dict that maps horizon --> algs_index -->  List[(HardwareSpec, embedding_index)]
    answer = dict()
    
    for (horizon, alg_index_d) in diff_algs_to_spec.items():
        assert horizon not in answer.keys()
        answer[horizon] = dict()
        for (alg_index, l) in alg_index_d.items():
            for (hardware_spec, embedding_index) in l:
                if hardware_spec not in answer[horizon].keys():
                    answer[horizon][hardware_spec] = dict()
                assert embedding_index not in answer[horizon][hardware_spec].keys()
                answer[horizon][hardware_spec][embedding_index] = alg_index
    return answer

def get_df_visualizing_lambdas(experiment_id, diff_algs_to_spec, take_best=True, mc_guarantees_file_name="mc_guarantees_.csv"):
    # diff_algs_to_spec is a dict that maps horizon --> algs_index -->  List[(HardwareSpec, embedding_index)]
    hard_spec_to_alg_index = get_hard_spec_to_alg_index(diff_algs_to_spec)
    hardware_specs_str = [] # strings that include the embedding
    hardwares = [] # just hardware (e.g. "Tenerife")
    horizons = []
    algorithm_types = [] # two possible values: "new" or "trad"
    algorithm_indices = [] 
    accs = []
    
    horizon_lines = dict() # maps horizon to (hardware_spec, min(acc, trad_acc), max(acc, trad_acc), alg_index)
    
    # load mc_guarantees file
    file_path = os.path.join(get_project_path(), get_experiment_name_path(experiment_id), experiment_id.value, mc_guarantees_file_name)
    f = open(file_path)
    lines = f.readlines()[1:]
    f.close()
    
    for line in lines:
        elements = line.split(",")
        hardware = elements[0]
        embedding_index = int(elements[1])
        horizon = int(elements[2])
        my_guarantee = float(elements[3])
        trad_guarantee = float(elements[4])
        best_trad_guarantee = float(elements[6])
        alg_index = hard_spec_to_alg_index[horizon][find_enum_object(hardware, HardwareSpec)][embedding_index] + 1
        
        
        if take_best:
            final_trad_guarantee = best_trad_guarantee
        else:
            final_trad_guarantee = trad_guarantee
        
        if my_guarantee != final_trad_guarantee:
            # only consider when we achieve something better than the traditional algorithm
            hardware_specs_str.append(f"{hardware}-{embedding_index}")
            hardwares.append(hardware)
            horizons.append(horizon)
            algorithm_types.append("new")
            algorithm_indices.append(alg_index)
            accs.append(my_guarantee)
            
            # horizon lines
            if horizon not in horizon_lines.keys():
                horizon_lines[horizon] = []
            
            horizon_lines[horizon].append((f"{hardware}-{embedding_index}", 
                                  min(my_guarantee, final_trad_guarantee), 
                                  max(my_guarantee, final_trad_guarantee),
                                  alg_index))
        
        hardware_specs_str.append(f"{hardware}-{embedding_index}")
        hardwares.append(hardware)
        horizons.append(horizon)
        algorithm_types.append("trad")
        algorithm_indices.append(0)
        accs.append(final_trad_guarantee)    

    df = pd.DataFrame.from_dict({
        'hardware_spec': hardware_specs_str,
        'hardware': hardwares,
        'horizon': horizons,
        'algorithm_type': algorithm_types,
        'alg_class': algorithm_indices,
        'accuracy': accs
    })
    return df, horizon_lines

def get_scatterplot_guarantees_compare(experiment_id, horizon, df, horizon_lines, horizon_to_num_algs, new_palette=None, hue_order=None):
    plt.clf()
    plt.figure(figsize=(15, 10))
    
    # Set Seaborn context
    sns.set(style="white", font_scale=1.5)

    if new_palette is None:
        new_palette = sns.color_palette(n_colors=horizon_to_num_algs[horizon]+1)
    if hue_order is None:
        hue_order = [x for x in range(0, horizon_to_num_algs[horizon]+1)]
    current_df = df[df.horizon==horizon]
    hardware_order = copy.deepcopy(df['hardware_spec'].values)
    for h in hardware_order:
        h_df = current_df[current_df.hardware_spec == h ]
        z_order = 4
        g = sns.scatterplot(data=h_df[h_df.algorithm_type == "new"], x="hardware_spec", y="accuracy", hue="alg_class", 
                            style="algorithm_type", legend=False, palette=new_palette,s=120, 
                            hue_order=hue_order, zorder=z_order, style_order=["trad", "new"])
        z_order = 3
        g = sns.scatterplot(data=h_df[h_df.algorithm_type == "trad"], x="hardware_spec", y="accuracy", hue="alg_class", 
                            style="algorithm_type", legend=False, palette=new_palette,s=120, 
                            hue_order=hue_order, zorder=z_order, style_order=["trad", "new"])

    if horizon in horizon_lines.keys():
        for line in horizon_lines[horizon]:
            x = line[0]
            ymin = line[1]
            ymax = line[2]
            color = new_palette[line[3]]
            g.vlines([x], ymin, ymax, colors=color, zorder=2)

    g.set(xticklabels=[])
    g.set(xlabel="hardware specification")
    # plt.ylim(0.45, 1.0001)
    output_dir = os.path.join(get_project_path(), get_experiment_name_path(experiment_id), experiment_id.value)
    plt.savefig(os.path.join(output_dir, f"scatter_guarantees{horizon}.pdf"), format='pdf')
    return g
