
from typing import List, Set
from graph import Graph, LowLevelGraph
from copy import deepcopy

from utils import QuantumChannel, Precision, default_mapping, myfloor, Instruction, AlgorithmNode, meas_instructions, get_inverse_mapping, meas_instructions, myceil, get_channel_algorithm_node
from cmath import isclose
from noise import get_ibm_noise_model

def get_obs_expected_val(graph, is_target_qs, address_space, obs):
    s = 0
    current_belief = graph.obs_to_belief[obs]
    for v in graph.observables_to_v[obs]:
        if is_target_qs(v.quantum_state, address_space):
            s += current_belief.get(v.id)
    return s
        # return myfloor(s, Precision.PRECISION)

def algorithm_with_tabs(alg, tabs):
    result_algorithm = []
    for line in alg:
        result_algorithm.append(f"{tabs}{line}")
    return result_algorithm
    
def get_bellman_algorithms(graph: Graph, is_target_qs, obs_to_algs, 
                               current_obs=0, inverse_mapping=None, address_space=default_mapping(), horizon=0):
    ''' obs_to_algs: store all optimal algorithms
    '''
    assert inverse_mapping is not None
    actions_to_expected_val = dict()
    actions_to_expected_val["I"] = get_obs_expected_val(graph, is_target_qs, address_space, current_obs)

    identity_node = AlgorithmNode(Instruction.I, 0,0)
    
    if horizon == 0:
        return (actions_to_expected_val["I"], [identity_node])
    actions_algs = dict()
    if (current_obs in graph.obs_adj_list.keys()) and (horizon > 0):
        for channel in graph.channels:
            if channel.name in graph.obs_adj_list[current_obs].keys():
                assert isinstance(channel, QuantumChannel)
                current_expected_val = 0
                current_algorithms = []
                if 0 in graph.obs_adj_list[current_obs][channel.name].keys():
                    post_obs0 = graph.obs_adj_list[current_obs][channel.name][0]
                    expected_val0, algs0 = get_bellman_algorithms(graph, is_target_qs, obs_to_algs, post_obs0,inverse_mapping=inverse_mapping, address_space=address_space, horizon=horizon-1)
                else:
                    expected_val0 = 0
                    algs0 = [None]
                if channel.get_instruction() in meas_instructions:
                    if 1 in graph.obs_adj_list[current_obs][channel.name].keys():
                        post_obs1 = graph.obs_adj_list[current_obs][channel.name][1]
                        expected_val1, algs1 = get_bellman_algorithms(graph, is_target_qs, obs_to_algs, post_obs1,inverse_mapping=inverse_mapping, address_space=address_space, horizon=horizon-1)
                    else:
                        expected_val1 = 0
                        algs1 = [None]
                    current_expected_val = expected_val0 + expected_val1
                    for alg0 in algs0:
                        for alg1 in algs1:
                            temp_head, temp_node = get_channel_algorithm_node(channel, inverse_mapping)
                            assert temp_node.next_ins is None
                            assert temp_node.case0 is None
                            assert temp_node.case1 is None
                            if alg0 is not None:
                                temp_node.case0 = deepcopy(alg0)
                            if alg1 is not None:
                                temp_node.case1 = deepcopy(alg1)
                            current_algorithms.append(temp_head)
                else:
                    current_expected_val = expected_val0
                    for alg0 in algs0:
                        temp_head, temp_node = get_channel_algorithm_node(channel, inverse_mapping)
                        assert temp_node.next_ins is None
                        assert temp_node.case0 is None
                        assert temp_node.case1 is None
                        if alg0 is not None:
                            temp_node.next_ins = deepcopy(alg0)
                        current_algorithms.append(temp_head)
                if current_expected_val > 1:
                    raise Exception(channel.name, current_expected_val)
                assert channel.name not in actions_to_expected_val.keys()
                assert channel.name not in actions_algs.keys()
                actions_to_expected_val[channel.name] = current_expected_val
                actions_algs[channel.name] = current_algorithms
    
    actions_algs["I"] = [identity_node]
    max_val = max(actions_to_expected_val.values())
    min_depth = -1 

    # compute the min depth among the algorithms with highest success rate (max_val)
    for (c, val) in actions_to_expected_val.items():
        if isclose(max_val, val, rel_tol=Precision.rel_tol):
            for a in actions_algs[c]:
                if min_depth == -1:
                    min_depth = a.depth
                else:
                    min_depth = min(min_depth, a.depth)
    
    # pick all the actions that lead to optimal values
    result_algorithms = []
    for (c, val) in actions_to_expected_val.items():
        if isclose(max_val, val, rel_tol=Precision.rel_tol):
            for a in actions_algs[c]:
                if a.depth == min_depth:
                    result_algorithms.append(a)
    obs_to_algs[current_obs] = (max_val, result_algorithms)
    return max_val, result_algorithms

def get_algorithm(current_node, tabs="\t", count_ifs = 0, for_ibm=False):
    if current_node is None:
        return f"{tabs}pass\n"
    assert isinstance(current_node, AlgorithmNode)
    if current_node.instruction == Instruction.I:
        assert current_node.next_ins is None
        if for_ibm:
            return f"{tabs}pass\n"
        return f"{tabs}return Result(qpu.count_executed_instructions, qpu.qmemory, qpu.meas_cache, qpu.log, Belief(), 0, qpu.instructions_applied, done=True, error=True)\n"
    
    if for_ibm:
        result = f"{tabs}instruction_to_ibm(qc, basis_gates, {current_node.instruction}, {current_node.target}, {current_node.control})\n"
    else:
        result = f"{tabs}outcome = qpu.apply_instructions([{current_node.instruction}], [address_space({current_node.target})] , [address_space({current_node.control})])\n"

    if current_node.instruction in meas_instructions:
        is_terminal = (current_node.case0 is None) and (current_node.case1 is None)
        if not is_terminal:
            if for_ibm:
                result += f"{tabs}with qc.if_test((cbits[{current_node.target}], 0)) as else{count_ifs}_:\n"
            else:
                result += f"{tabs}if outcome == 0:\n"
            alg0 = get_algorithm(current_node.case0, tabs= f"{tabs}\t", count_ifs=count_ifs+1, for_ibm=for_ibm)
            result += alg0
            alg1 = get_algorithm(current_node.case1, tabs=f"{tabs}\t", count_ifs=count_ifs+1, for_ibm=for_ibm)
            if for_ibm:
                result += f"{tabs}with else{count_ifs}_:\n"
            else:
                result += f"{tabs}else:\n"
            result += alg1
    else:
        is_terminal = current_node.next_ins is None
        if not is_terminal:
            next_algorithm = get_algorithm(current_node.next_ins, tabs, count_ifs, for_ibm=for_ibm)
            result += next_algorithm
    return result

def dump_algorithms(algorithms: List[AlgorithmNode], output_path, for_ibm=False, comments=None):
    file = open(output_path, "w")
    file.write("import os, sys\n")
    file.write("sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n")
    if for_ibm:
        file.write("from qiskit import QuantumCircuit, ClassicalRegister\n")
        file.write("from utils import Instruction, instruction_to_ibm\n\n")
    else:
        file.write("from utils import *\n")
        file.write("from simulator import *\n\n")
    assert (comments is None) or (len(comments) == len(algorithms))

    for (index, algorithm) in enumerate(algorithms):
        assert isinstance(algorithm, AlgorithmNode)
        if for_ibm:
            file.write(f"def algorithm{index}(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):\n")
        else:
            file.write(f"def algorithm{index}(qpu, address_space=default_mapping()):\n")
        if comments is not None:
            initial_comment = comments[index]
            file.write(f"\t\'\'\'{initial_comment}\'\'\'\n")
        file.write(get_algorithm(algorithm, for_ibm=for_ibm))
        file.write("\n\n")

    file.write("algorithms = []\n")
    for i in range(len(algorithms)):
        file.write(f"algorithms.append(algorithm{i})\n")
    file.close()

def serialize_algorithms(algorithms: List[AlgorithmNode], output_path):
    assert isinstance(output_path, str)
    file = open(output_path, "w")
    file.write("import os, sys\n")
    file.write("sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n")
    file.write("from utils import *\n\n")
    file.write("algorithms = [")
    for (index, algorithm) in enumerate(algorithms):
        assert isinstance(algorithm, AlgorithmNode)
        file.write(f"{algorithm.serialize()} \n")
        if index + 1 != len(algorithms):
            file.write(",")
        elif index == 0:
            file.write("\n")
    file.write("]")
    file.close()

def get_bellman_graph(graph: Graph, is_target_qs, address_space=default_mapping(), outpath=None, horizon=6) -> str:
    obs_to_algs = dict()
    inverse_mapping = get_inverse_mapping(address_space)
    expected_val, algorithms = get_bellman_algorithms(graph, is_target_qs, obs_to_algs, inverse_mapping=inverse_mapping, address_space=address_space, horizon=horizon)
    if outpath is not None:
        assert isinstance(outpath, str)
        serialize_algorithms(algorithms, outpath)
    if Precision.is_lowerbound:
        return float(myfloor(float(expected_val), Precision.PRECISION))
    else:
        return float(myceil(float(expected_val), Precision.PRECISION))

def get_real_accuracy(algorithm_node, backend, initial_states, 
                      is_target_qs, address_space, horizon, path_prefix="", precision=30):
    assert isinstance(backend, str)
    previous_prec = Precision.PRECISION
    Precision.PRECISION = precision
    Precision.update_threshold()
    noise_model = get_ibm_noise_model(backend, path_prefix=path_prefix)
    assert isinstance(algorithm_node, AlgorithmNode)
    instructions_needed = [] 
    algorithm_node.get_instructions_used(instructions_needed)

    channels = []

    for (instruction, target, control) in instructions_needed:
        if instruction in meas_instructions:
            assert control is None
            c = noise_model.get_meas_channel(instruction, address_space[target])
        else:
            if control is None:
                control_ = None
            else:
                control_ = address_space[control]
            c = noise_model.get_instruction_channel(address_space[target], instruction, control=control_)
        channels.append(c)

    g = Graph(channels=channels, initial_states=initial_states)
    g.build_graph_from_algorithm(algorithm_node)
    real_acc = get_bellman_graph(g, is_target_qs, outpath=None, address_space=address_space, horizon=horizon)
    Precision.PRECISION = previous_prec
    Precision.update_threshold()
    return real_acc

    
