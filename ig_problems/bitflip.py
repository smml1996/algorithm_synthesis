import os, sys
from typing import List

sys.path.append(os.getcwd()+"/..")

from utils import Precision
from experiments_utils import get_default_embedding, get_project_path
from qpu_utils import Op
from ibm_noise_models import Instruction, NoiseModel
from pomdp import POMDPAction, build_pomdp
from problems.bitflip import MAX_PRECISION, BitFlipInstance, bitflips_guard

from imperfect_games import KnwGraph


class IGBitflipInstance(BitFlipInstance):
    def __init__(self, embedding):
        super().__init__(embedding)
        
    def is_target_qs(self, hybrid_state):
        return super().get_reward(hybrid_state) > 0.0
    

def get_experiments_actions(embedding)-> List[POMDPAction]:
    X0 = POMDPAction("X0", [Instruction(embedding[0], Op.X)])
    CX = POMDPAction("CX", [Instruction(embedding[2], Op.CNOT, control=embedding[0]), Instruction(embedding[2], Op.CNOT, control=embedding[1])])
    P2 = POMDPAction("P2", [Instruction(embedding[2], Op.MEAS)])
    return [X0, CX, P2]

if __name__ == "__main__":
    Precision.PRECISION = MAX_PRECISION
    Precision.update_threshold()
    
    max_horizon = 10000
    noise_model = NoiseModel()
    
    embedding = get_default_embedding(3)
    problem_instance = IGBitflipInstance(embedding)
    actions = get_experiments_actions(embedding)
    initial_distribution = []
    
    for initial_state in problem_instance.initial_state:
        initial_distribution.append((initial_state, 1/4))
    
    pomdp = build_pomdp(actions, noise_model, max_horizon, embedding, initial_distribution=initial_distribution, guard=bitflips_guard)
    pomdp.serialize(problem_instance, "pomdp.txt")
    
    knwgraph = KnwGraph(pomdp, problem_instance.is_target_qs)
    algorithm = knwgraph.get_algorithm()
    
    project_path = get_project_path()
    output_path = os.path.join(project_path, "results", "ig_bitflip", "algorithm.py")
    algorithm.dump(output_path, actions)
    
    algorithm.check(problem_instance.initial_state, problem_instance.is_target_qs)

