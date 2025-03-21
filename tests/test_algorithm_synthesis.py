import os, sys
sys.path.append(os.getcwd()+"/..")

from algorithm import dump_algorithms
from experiments_utils import get_bellman_value, get_ibm_simulated_acc, get_markov_chain_results, get_project_path
from pomdp import build_pomdp
from problems.bitflip import *

experiment_id = BitflipExperimentID.IPMA2
hardware_spec = HardwareSpec.ATHENS
embedding = {0: 0, 1: 2, 2: 1}
project_path = get_project_path()
current_path = os.path.join(project_path, "tests")
pomdp_path = os.path.join(current_path, "test_pomdp.txt")
algorithm_path = os.path.join(current_path, "test_algorithm.json")
horizon = 3
def test_bitflip_synthesis():
    hardware_spec = HardwareSpec.ATHENS
    noise_model = NoiseModel(hardware_spec, thermal_relaxation=WITH_TERMALIZATION)
    actions = get_experiments_actions(noise_model, embedding, experiment_id)
    problem_instance = BitFlipInstance(embedding)
    
    pomdp = build_pomdp(actions, noise_model, horizon, embedding, initial_distribution=problem_instance.get_initial_distribution(), guard=bitflips_guard, qubits_used=embedding.values())
    
    
    pomdp_path = os.path.join(current_path, "test_pomdp.txt")
    algorithm_path = os.path.join(current_path, "test_algorithm.json")
    pomdp.serialize(problem_instance, pomdp_path)
    
    bellman_val =  get_bellman_value(get_project_settings(), pomdp_path, from_pomdp_path=True, horizon=horizon, output_path=algorithm_path)
    
    # assert abs(bellman_val-real_val) < 0.01
    mk_result = get_markov_chain_results(get_project_settings(), algorithm_path, pomdp_path)
    assert mk_result == bellman_val
    
def dump_algorithm_test():
    noise_model = NoiseModel(hardware_spec, thermal_relaxation=WITH_TERMALIZATION)
    actions_to_instructions = dict()
    actions = get_experiments_actions(noise_model, {0:0, 1:1, 2:2}, experiment_id)
    for action in actions:
        actions_to_instructions[action.name] = action.instruction_sequence
    actions_to_instructions["halt"] = []
    algorithm = AlgorithmNode(serialized=json.load(open(algorithm_path)), actions_to_instructions=actions_to_instructions)
    dump_algorithms([algorithm], actions, os.path.join("algorithm_ibm.py"))
    
    
def simulation_test():
    mk_result = get_markov_chain_results(get_project_settings(), algorithm_path, pomdp_path)
    
    # simulate algorithm in IBM simulator
    noise_model = NoiseModel(hardware_spec, thermal_relaxation=WITH_TERMALIZATION)
    actions_to_instructions = dict()
    actions = get_experiments_actions(noise_model, {0:0, 1:1, 2:2}, experiment_id)
    for action in actions:
        actions_to_instructions[action.name] = action.instruction_sequence
    actions_to_instructions["halt"] = []
    algorithm = AlgorithmNode(serialized=json.load(open(algorithm_path)), actions_to_instructions=actions_to_instructions)
    simulated_result = get_ibm_simulated_acc(algorithm, embedding, hardware_spec, IBMBitFlipInstance, init_states=[0,1,2,3])
    assert isclose(mk_result, simulated_result, rel_tol=1e-3)
    
    


if __name__ == "__main__":
    Precision.PRECISION = MAX_PRECISION
    Precision.update_threshold()
   
    # test_bitflip_synthesis()
    dump_algorithm_test()
    # simulation_test()