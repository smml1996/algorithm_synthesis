import os, sys
sys.path.append(os.getcwd()+"/..")

from copy import deepcopy
from typing import Dict, List


from cmemory import ClassicalState
from pomdp import POMDPAction
import qmemory
from qpu_utils import GateData, Op, BasisGates
from utils import Precision
sys.path.append(os.getcwd()+"/..")

from qstates import QuantumState
from ibm_noise_models import Instruction, MeasChannel, NoiseModel, HardwareSpec
import numpy as np
from math import pi   
from experiments_utils import ReadoutNoise, ResetExperimentID, bitflips_guard, generate_configs, generate_diff_algorithms_file, generate_embeddings, generate_mc_guarantees_file, generate_pomdps, get_config_path, get_num_qubits_to_hardware, get_project_settings



WITH_TERMALIZATION = False
MAX_PRECISION = 10
TIME_OUT = 10800 # (in seconds) i.e 3 hours


class ResetInstance:
    def __init__(self, embedding):
        """_summary_

        Args:
            num_qubits (int): _description_
            instruction_set (List[Instruction]): _description_
            embedding (Dict[int, int]): a mapping from logical qubits to physical qubits
        """
        self.embedding = embedding
        self.initial_state = None
        self.get_initial_states()
        # check embedding
        assert 0 in self.embedding.keys()
        assert len(self.embedding.keys()) == 1

    def get_initial_states(self):
        """
        The initial state is specified as a uniform superpositions over all four bell states.
        """
        self.initial_state = []
        initial_cs = ClassicalState()

        X0 = Instruction(self.embedding[0], Op.X).get_gate_data()

        # prepare first bell state
        state0 = QuantumState(0, qubits_used=list(self.embedding.values()))
        self.initial_state.append((state0, initial_cs))

        # prepare second bell state
        state1 = qmemory.handle_write(state0, X0)
        self.initial_state.append((state1, initial_cs))


    def get_reward(self, hybrid_state) -> float:
        state0 = QuantumState(0, qubits_used=list(self.embedding.values()))
        qs , _ = hybrid_state
        if qs == state0:
            return 1.00
        else:
            return 0.00
    

# choosing embeddings
def get_pivot_qubits(noise_model: NoiseModel, experiment_id: ResetExperimentID):
    result = set()
    noises = []
    if noise_model.hardware_spec == HardwareSpec.MELBOURNE:
        noise_model.num_qubits = 14
    for qubit in range(noise_model.num_qubits):
        assert experiment_id == ResetExperimentID.main
        if noise_model.get_qubit_outdegree(qubit) > 1:
            noise_data = noise_model.instructions_to_channel[Instruction(qubit, Op.MEAS)]
            assert isinstance(noise_data, MeasChannel)
            success0 = noise_data.get_ind_probability(0,0)
            success1 = noise_data.get_ind_probability(1,1)
            noises.append(ReadoutNoise(qubit, success0, success1))

    temp = sorted(noises, key=lambda x : x.success0)
    result.add(temp[0].target)

    temp = sorted(noises, key=lambda x : x.success1)
    result.add(temp[0].target)

    temp = sorted(noises, key=lambda x: x.acc_err) # accumulated error
    result.add(temp[0].target)

    temp = sorted(noises, key=lambda x: x.diff)
    if temp[0].diff != temp[len(temp)-1].diff:
        result.add(temp[0].target)

    temp = sorted(noises, key=lambda x: x.abs_diff)
    if temp[0].abs_diff != temp[len(temp)-1].abs_diff:
        result.add(temp[0].target)
        assert (temp[0].abs_diff < temp[len(temp)-1].abs_diff)
    return result

def get_hardware_embeddings(backend: HardwareSpec, experiment_id) -> List[Dict[int, int]]:
    result = []
    parsed_pivots = set()
    noise_model = NoiseModel(backend, thermal_relaxation=WITH_TERMALIZATION)
    assert noise_model.num_qubits >= 14
    pivot_qubits = get_pivot_qubits(noise_model, experiment_id)
    for target in pivot_qubits:
        assert(isinstance(target, int))
        assert target not in parsed_pivots
        d_temp = dict()
        d_temp[0] = target
        result.append(deepcopy(d_temp))
        parsed_pivots.add(target)
    return result

def get_experiments_actions(noise_model: NoiseModel, embedding: Dict[int,int], experiment_id: ResetExperimentID):
    if experiment_id == ResetExperimentID.main:
        if noise_model.basis_gates in [BasisGates.TYPE1]:
            X0 = POMDPAction("X0", [Instruction(embedding[0], Op.U3, params=[pi, 2*pi, pi])])
        else:
            X0 = POMDPAction("X0", [Instruction(embedding[0], Op.X)])
            
        P0 = POMDPAction("P2", [Instruction(embedding[0], Op.MEAS)])
        return [P0, X0]
    else:
        raise Exception(f"Invalid experiment id {experiment_id}")


if __name__ == "__main__":
    arg_backend = sys.argv[1]
    Precision.PRECISION = MAX_PRECISION
    Precision.update_threshold()
    settings = get_project_settings()
    project_path = settings["PROJECT_PATH"]
    
    allowed_hardware = []
    for hardware in HardwareSpec:
        noise_model = NoiseModel(hardware, thermal_relaxation=WITH_TERMALIZATION)
        if noise_model.num_qubits >= 14:
            allowed_hardware.append(hardware)
    if arg_backend == "gen_configs":
        # step 0
        generate_configs(experiment_id=ResetExperimentID.main, min_horizon=2, max_horizon=7, allowed_hardware=allowed_hardware)
    elif arg_backend == "embeddings":
        # generate paper embeddings
        batches = get_num_qubits_to_hardware(WITH_TERMALIZATION, allowed_hardware=allowed_hardware)
        for num_qubits in batches.keys():
            reset_config_path = get_config_path(ResetExperimentID.main, num_qubits)
            generate_embeddings(ResetExperimentID.main, num_qubits, get_hardware_embeddings)
    elif arg_backend == "all_pomdps":
        # TODO: clean me up
        # step 2: generate all pomdps
        # config_path = sys.argv[2]
        # generate_pomdps(config_path)
        batches = get_num_qubits_to_hardware(WITH_TERMALIZATION, allowed_hardware=allowed_hardware)
        for num_qubits in batches.keys():
            generate_pomdps(ResetExperimentID.main, num_qubits, get_experiments_actions, ResetInstance)
    elif arg_backend == "mc_ipma":
        generate_mc_guarantees_file(PhaseflipExperimentID.IPMA, allowed_hardware, get_hardware_embeddings, get_experiments_actions, WITH_THERMALIZATION=WITH_TERMALIZATION) 
    elif arg_backend == "alg_ipma":
        generate_diff_algorithms_file(PhaseflipExperimentID.IPMA, allowed_hardware, get_hardware_embeddings, get_experiments_actions, with_thermalization=False)
        
    # step 3 synthesis of algorithms with C++ code and generate lambdas (guarantees)
    
        