
from argparse import Action
from cmath import isclose, pi
from enum import Enum
import time
import numpy as np
import json
import os, sys
sys.path.append(os.getcwd() + "/..")

from experiments_utils import default_load_embeddings, directory_exists, generate_configs, generate_embeddings, get_bellman_value, get_config_path, get_project_settings
from pomdp import POMDPAction, build_pomdp
import qmemory
from qpu_utils import BasisGates, Op


from ibm_noise_models import HardwareSpec, Instruction, NoiseModel, get_num_qubits_to_hardware, load_config_file
from typing import Any, Dict, List
from utils import Precision, find_enum_object, np_get_ground_state
from cmemory import ClassicalState
from qstates import QuantumState, np_get_energy, np_get_fidelity, np_schroedinger_equation
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature import settings

settings.use_pauli_sum_op = False

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter

from scipy.optimize import minimize

MAX_PRECISION = 10
WITH_THERMALIZATION = False
P0_ALLOWED_HARDWARE = [HardwareSpec.AUCKLAND, HardwareSpec.WASHINGTON, HardwareSpec.ROCHESTER]


class ParamInsExperimentId(Enum):
    H2Mol_Q1 = "H2Mol_Q1"
    

def get_hamiltonian() -> SparsePauliOp:
    # Define the hydrogen molecule at internuclear distance R = 0.75 angstroms
    molecule = MoleculeInfo(
        ["H", "H"], 
        [(0.0, 0.0, 0.0), (0.0, 0.0, 0.75)],
        charge=0,
        multiplicity=1,
        units=DistanceUnit.ANGSTROM
    )
    
    # Set up the PySCF driver to calculate molecular integrals
    driver = PySCFDriver.from_molecule(molecule=molecule, basis="sto3g")

    # Obtain the electronic structure problem
    problem = driver.run()
    
    hamiltonian = problem.hamiltonian.second_q_op()
    mapper = JordanWignerMapper()
    converter = QubitConverter(mapper, z2symmetry_reduction="auto")
    
    qubit_op = converter.convert(
        hamiltonian,
        num_particles=problem.num_particles,
        sector_locator=problem.symmetry_sector_locator,
    )

    assert isinstance(qubit_op, SparsePauliOp)
    return qubit_op

    
class ParamInsInstance:
    def __init__(self, embedding, experiment_id) -> None:
        self.experiment_id = experiment_id
        self.embedding = embedding
        self.initial_state = None
        self.get_initial_states()
        if experiment_id == ParamInsExperimentId.H2Mol_Q1:
            self.hamiltonian = get_hamiltonian()
         
    def get_initial_states(self):
        if self.experiment_id in [ParamInsExperimentId.H2Mol_Q1]:
            quantum_state = QuantumState(0, qubits_used=list(self.embedding.values()))
            X0 = Instruction(self.embedding[0], Op.X).get_gate_data()
            quantum_state = qmemory.handle_write(quantum_state, X0)
        assert quantum_state is not None
        self.initial_state = (quantum_state, ClassicalState())
        
    def get_reward(self, hybrid_state) -> bool:
        qs, cs = hybrid_state
        assert isinstance(qs, QuantumState)
        if self.experiment_id in [ParamInsExperimentId.H2Mol_Q1]:
            return np_get_energy(self.hamiltonian, qs.to_np_array())
        
def get_actions(noise_model: NoiseModel, embedding: Dict[int,int], experiment_id: ParamInsExperimentId) -> List[Action]:
   
    if experiment_id == ParamInsExperimentId.H2Mol_Q1:
        U3_gate = Instruction(embedding[0], Op.U3, params=['a', 'b', 'c'], symbols=['a', 'b', 'c']).to_basis_gate_impl(noise_model.basis_gates)
        return [POMDPAction("U3", U3_gate)]
    else:
        raise Exception("Not implemented!")

def get_hardware_embeddings(hardware: HardwareSpec, **kwargs) -> List[Dict[int, int]]:
    noise_model = NoiseModel(hardware, thermal_relaxation=WITH_THERMALIZATION)
    # assert hardware not in kwargs["statistics"].keys()
    if "statistics" in kwargs.keys():
        kwargs["statistics"][hardware] = []
    if kwargs["experiment_id"] in [ParamInsExperimentId.H2Mol_Q1]:
        answer = []
        pivot_qubits = set()

        # get qubit with highest accumulated measurement error rate
        if noise_model.basis_gates in [BasisGates.TYPE1, BasisGates.TYPE6]:
            # we get the most noisy qubits in terms of U1 and U2
            for reverse in [False]:
                most_noisy_U3 = noise_model.get_most_noisy_qubit(Op.U3, reverse=reverse)[0]
                if "statistics" in kwargs.keys():
                    kwargs["statistics"][hardware].append((most_noisy_U3, Op.U3, not reverse))
                pivot_qubits.add(most_noisy_U3[1])
            
        else:
            for reverse in [False]:
                # for some reason, IBM does not has error models for RZ gates
                # we get the most noisy qubits in terms of SX and RZ gates
                most_noisy_SX = noise_model.get_most_noisy_qubit(Op.SX, reverse=reverse)[0]
                # most_noisy_RZ = noise_model.get_most_noisy_qubit(Op.RZ)[0]
                if "statistics" in kwargs.keys():
                    kwargs["statistics"][hardware].append((most_noisy_SX, Op.SX, not reverse))
                # kwargs["statistics"][hardware].append((most_noisy_RZ, Op.RZ))
                pivot_qubits.add(most_noisy_SX[1])
                # pivot_qubits.add(most_noisy_RZ[1])
            
        for p in pivot_qubits:
            answer.append({0: p})
        return answer
    else:
        raise Exception("Not implemented!")

def get_experiment_batches():
    batches = dict()
    for hardware in P0_ALLOWED_HARDWARE:
        batches[hardware.value] = [hardware.value]
    return batches

def get_initial_parameters(config) -> List[float]:
    if config["experiment_id"] == ParamInsExperimentId.H2Mol_Q1:
        return [0.0, 0.0, 0.0]
    raise Exception(f"Initial parameters for experiment {config['experiment_id']} not specified")

def cost_function(params: List[float], noise_model: NoiseModel, parametric_actions: List[POMDPAction], config: Dict[Any, Any], problem_instance: ParamInsInstance, energy_history: List[float], project_settings: Dict[str, str], config_path: str):
    horizon = config["max_horizon"]
    
    hardware_str = config["hardware"][0]
    output_path = os.path.join(config["output_dir"], "pomdps", f"{hardware_str}_0.txt")
    
    # bind params
    bind_dict = dict()
    current_param_index = 0
    for parametric_action in parametric_actions:
        print("paramatric action", parametric_action.symbols)
        for symbol in parametric_action.symbols:
            if symbol not in bind_dict.keys():
                bind_dict[symbol] = params[current_param_index]
                current_param_index += 1
    assert current_param_index == len(params)
    actions = []
    for parametric_action in parametric_actions:
        actions.append(parametric_action.bind_symbols_from_dict(bind_dict))
    
    pomdp = build_pomdp(actions, noise_model, horizon, problem_instance.embedding, initial_state=problem_instance.initial_state)
    # pomdp.optimize_graph(problem_instance) # no optimization because every vertex has its own energy
    
    # save POMDP
    pomdp.serialize(problem_instance, output_path) 
    
    # now that pomdp is ready we must run bellman equation
    energy = get_bellman_value(project_settings, config_path)
    energy_history.append(energy)
    return energy
    

if __name__ == "__main__":
    arg_backend = sys.argv[1]
    Precision.PRECISION = MAX_PRECISION
    Precision.update_threshold()
    
    if arg_backend == "gen_configs":
        generate_configs("param_ins", ParamInsExperimentId.H2Mol_Q1, 1, 2, allowed_hardware=P0_ALLOWED_HARDWARE, batches=get_experiment_batches(), opt_technique="min")
        
    if arg_backend == "embeddings":
        batches = get_experiment_batches()
        for batch_name in batches.keys():
            config_path = get_config_path("param_ins", ParamInsExperimentId.H2Mol_Q1, batch_name)
            generate_embeddings(config_path=config_path, experiment_enum=ParamInsExperimentId, experiment_id=ParamInsExperimentId.H2Mol_Q1, get_hardware_embeddings=get_hardware_embeddings)
            
    if arg_backend == "run_config":
        config_path = sys.argv[2]
        config = load_config_file(config_path, ParamInsExperimentId)
        directory_exists(config["output_dir"])
        
        output_folder = os.path.join(config["output_dir"], "pomdps")
        # check that there is a folder with the experiment id inside pomdps path
        directory_exists(output_folder)
        
        assert len(config["hardware"]) == 1
        hardware_spec = find_enum_object(config["hardware"][0], HardwareSpec)
        noise_model = NoiseModel(hardware_spec, thermal_relaxation=WITH_THERMALIZATION)
        initial_parameters = get_initial_parameters(config)
        
        embeddings = default_load_embeddings(config, ParamInsExperimentId)
        print(embeddings)
        assert embeddings["count"] == 1
        embeddings = embeddings[hardware_spec]["embeddings"]
        assert len(embeddings) == 1
        embedding = embeddings[0]
        
        actions = get_actions(noise_model, embedding, ParamInsExperimentId.H2Mol_Q1)
        for action in actions:
            print("action", action.symbols)
        
        problem_instance = ParamInsInstance(embedding, ParamInsExperimentId.H2Mol_Q1)
        
        project_settings = get_project_settings()
        energy_history = []
        result = minimize(cost_function, initial_parameters, 
                        args=(noise_model, actions, config, problem_instance, energy_history, project_settings, config_path), 
                        method="SLSQP")
        
        
        
       