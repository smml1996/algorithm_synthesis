import os, sys
sys.path.append(os.getcwd()+"/..")

from experiments_utils import get_bellman_value, get_markov_chain_results, get_project_path
from pomdp import build_pomdp
from problems.bitflip import *

def test_bitflip_synthesis():
    real_val = 0.9817750883276146644
    horizon = 3
    project_path = get_project_path()
    current_path = os.path.join(project_path, "tests")
    experiment_id = BitflipExperimentID.IPMA2
    hardware_spec = HardwareSpec.ATHENS
    noise_model = NoiseModel(hardware_spec, thermal_relaxation=WITH_TERMALIZATION)
    embedding = {0: 0, 1: 2, 2: 1}
    actions = get_experiments_actions(noise_model, embedding, experiment_id)
    problem_instance = BitFlipInstance(embedding)
    
    pomdp = build_pomdp(actions, noise_model, horizon, embedding, initial_distribution=problem_instance.get_initial_distribution(), guard=bitflips_guard, qubits_used=embedding.values())
    
    
    pomdp_path = os.path.join(current_path, "test_pomdp.txt")
    algorithm_path = os.path.join(current_path, "test_algorithm.json")
    pomdp.serialize(problem_instance, pomdp_path)
    
    bellman_val =  get_bellman_value(get_project_settings(), pomdp_path, from_pomdp_path=True, horizon=horizon, output_path=algorithm_path)
    print("bellman val:",bellman_val)
    
    # assert abs(bellman_val-real_val) < 0.01
    mk_result = get_markov_chain_results(get_project_settings(), algorithm_path, pomdp_path)
    print("mk result:", mk_result)
    


if __name__ == "__main__":
    Precision.PRECISION = MAX_PRECISION
    Precision.update_threshold()
    test_bitflip_synthesis()