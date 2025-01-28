from ghz import *

noise_model = NoiseModel(HardwareSpec.ALMADEN, thermal_relaxation=False)
embedding = {0: 10, 1: 5, 2: 11}
actions = get_experiments_actions(noise_model, embedding, GHZExperimentID.EXP1)
for action in actions:
    print(action.instruction_sequence)