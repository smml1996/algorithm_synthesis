
from argparse import Action
from cmath import isclose, pi
from enum import Enum
import time
import numpy as np
import json
import os, sys


from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
sys.path.append(os.getcwd() + "/..")

from algorithm import AlgorithmNode, execute_algorithm
from experiments_utils import default_load_embeddings, directory_exists, generate_configs, generate_embeddings, generate_embeddings_files, get_bellman_value, get_config_path, get_markov_chain_results, get_project_settings
from pomdp import POMDPAction, build_pomdp
import qmemory
from qpu_utils import BasisGates, Op


from ibm_noise_models import HardwareSpec, Instruction, NoiseModel, get_ibm_noise_model, get_num_qubits_to_hardware, ibm_simulate_circuit, instruction_to_ibm, load_config_file
from typing import Any, Dict, List
from utils import Precision, find_enum_object, np_get_ground_state
from cmemory import ClassicalState
from qstates import QuantumState, np_array_to_qs, np_get_energy, np_get_energy_from_rho, np_get_fidelity, np_schroedinger_equation
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature import settings
from qiskit.primitives import Estimator
from qiskit.circuit import Parameter

settings.use_pauli_sum_op = False

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter, ParityMapper
from qiskit.circuit.library import EfficientSU2

from scipy.optimize import minimize

MAX_PRECISION = 10
WITH_THERMALIZATION = False
P0_ALLOWED_HARDWARE = [HardwareSpec.AUCKLAND, HardwareSpec.WASHINGTON, HardwareSpec.ROCHESTER]


MINIMIZATION_METHODS = ["SLSQP", #Gradient-based method. Constrained, smooth problems.
                        "BFGS", # gradient-based. Smooth, unconstrained problems.
                        "Powell", # Derivative-free method. Functions that are discontinuous or noisy.
                        "CG" # Gradient-based method.  Large, smooth, unconstrained problems.
                        ]

class ParamInsExperimentId(Enum):
    H2Mol_Q1 = "H2Mol_Q1"
    # real experiments:
    H2Mol_Q1_SU2_Min = "H2Mol_Q1_SU2_Min" # Tries Efficient SU(2) gates(Rx and Ry), minimizes energy
    H2Mol_Q1_SU2_Max = "H2Mol_Q1_SU2_Max" # Tries Efficient SU(2) gates(Rx and Ry), maximizes probability of reaching ground state
    H2Mol_Q2_An_SU2_Min = "H2Mol_Q2_An_SU2_Min" # Efficient SU(2) gates(Rx and Ry CX), minimizes energy
    H2Mol_Q2_An_SU2_Max = "H2Mol_Q2_An_SU2_Max" # Efficient SU(2) gates(Rx and Ry CX), maximizes probability of reaching ground state. Second qubit is an ancilla an it include measurements.
    H2Mol_Q2_SU2_Min = "H2Mol_Q2_SU2_Min" # Efficient SU(2) gates(Rx and Ry CX), minimizes energy
    H2Mol_Q2_SU2_Max = "H2Mol_Q2_SU2_Max" # Efficient SU(2) gates(Rx and Ry CX), maximizes probability of reaching ground state. Second qubit is an ancilla an it include measurements.
    
    

def get_hamiltonian(experiment_id: ParamInsExperimentId) -> SparsePauliOp:
    # https://qiskit-community.github.io/qiskit-nature/migration/0.6_c_qubit_converter.html
    if experiment_id in [ParamInsExperimentId.H2Mol_Q1, ParamInsExperimentId.H2Mol_Q1_SU2_Min, ParamInsExperimentId.H2Mol_Q1_SU2_Max, ParamInsExperimentId.H2Mol_Q2_An_SU2_Max, ParamInsExperimentId.H2Mol_Q2_An_SU2_Min]:
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
    elif experiment_id in [ParamInsExperimentId.H2Mol_Q2_SU2_Min, ParamInsExperimentId.H2Mol_Q2_SU2_Max]:
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
        mapper = ParityMapper(num_particles=problem.num_particles)
        reduced_op = mapper.map(hamiltonian)

        assert isinstance(reduced_op, SparsePauliOp)
        return reduced_op

    
class ParamInsInstance:
    def __init__(self, embedding: Dict[int, int], experiment_id: ParamInsExperimentId, num_parameters:int , reps=0) -> None:
        self.experiment_id = experiment_id
        self.embedding = embedding
        self.initial_state = None
        self.get_initial_states()
        self.hamiltonian = get_hamiltonian(experiment_id)
        self.reps=reps
        self.initial_parameters = [0.0 for _ in range(num_parameters)]
        if experiment_id in [ParamInsExperimentId.H2Mol_Q1, ParamInsExperimentId.H2Mol_Q1_SU2_Max, ParamInsExperimentId.H2Mol_Q2_An_SU2_Max, ParamInsExperimentId.H2Mol_Q2_SU2_Max]:
            # Here we compute the ground energy which is mainly needed for finding the probability of reaching the ground state in certain experiments
            if experiment_id == ParamInsExperimentId.H2Mol_Q2_An_SU2_Max:
                initial_qs = QuantumState(0, qubits_used=[0])
                X0 = Instruction(0, Op.X).get_gate_data()
                initial_qs = qmemory.handle_write(initial_qs, X0)
            else:
                initial_qs = self.initial_state[0]
                
            target_state = initial_qs
            self.target_energy = np_get_energy(self.hamiltonian, target_state)
            while True:
                target_state = np_schroedinger_equation(self.hamiltonian, complex(0, -14), initial_qs.to_np_array())
                new_energy = np_get_energy(self.hamiltonian, target_state)
                if isclose(new_energy, self.target_energy, abs_tol=Precision.isclose_abstol):
                    break
                self.target_energy = new_energy
        
         
    def get_initial_states(self):
        # the initial state is with all qubits in state |1>
        quantum_state = QuantumState(0, qubits_used=list(self.embedding.values()))
        for key in self.embedding.keys():
            X0 = Instruction(self.embedding[key], Op.X).get_gate_data()
            quantum_state = qmemory.handle_write(quantum_state, X0)
            assert quantum_state is not None
        self.initial_state = (quantum_state, ClassicalState())
        
    def get_ibm_initial_state(self, qc: QuantumCircuit, basis_gates: BasisGates):
        # the initial state is with all qubits in state |1>
        for key in self.embedding.keys():
            instruction = Instruction(key, Op.X)
            instruction_seq = instruction.to_basis_gate_impl(basis_gates)
            instruction_to_ibm(qc, instruction_seq, noiseless=True)
            
    def get_ibm_ansatz(self, qc: QuantumCircuit):
        if self.experiment_id in [ParamInsExperimentId.H2Mol_Q1]:
            a = Parameter('a')
            b = Parameter('b')
            c = Parameter('c')
            
            qc.u(a,b,c, 0) # applies u3 gate to qubit 0
        if self.experiment_id in [ParamInsInstance.H2Mol_Q1_SU2_Min, ParamInsInstance.H2Mol_Q1_SU2_Max, ParamInsInstance.H2Mol_Q2_An_SU2_Min, ParamInsInstance.H2Mol_Q2_An_SU2_Max]:
            ansatz = EfficientSU2(len(self.embedding.keys()), su2_gates=["rx", "ry"], entanglement="linear", reps=self.reps)
            qc.compose(ansatz, inplace=True)
        else:
            assert self.experiment_id in [ParamInsExperimentId.H2Mol_Q2_SU2_Min, ParamInsExperimentId.H2Mol_Q2_SU2_Max]
            ansatz = EfficientSU2(len(self.embedding.keys()), su2_gates=["rx", "ry"], entanglement="linear", reps=self.reps)
            qc.compose(ansatz, inplace=True)
            
    def get_ibm_ansatz_actions(self, noise_model: NoiseModel) -> List[POMDPAction]:
        if self.experiment_id in [ParamInsExperimentId.H2Mol_Q1_SU2_Min, ParamInsExperimentId.H2Mol_Q1_SU2_Max, ParamInsExperimentId.H2Mol_Q2_SU2_Max, ParamInsExperimentId.H2Mol_Q2_SU2_Min]:
            return get_actions(noise_model, self.embedding, self.experiment_id, reps)
        else:
            assert self.experiment_id in [ParamInsExperimentId.H2Mol_Q2_An_SU2_Min, ParamInsExperimentId.H2Mol_Q2_An_SU2_Max]
            return get_actions(noise_model, self.embedding, ParamInsExperimentId.H2Mol_Q2_SU2_Max, reps)
    
    def get_ibm_horizon(self) -> int:
        if self.experiment_id in [ParamInsExperimentId.H2Mol_Q1_SU2_Min, ParamInsExperimentId.H2Mol_Q1_SU2_Max]:
            return 2 * (self.reps+1)
        else:
            assert self.experiment_id in [ParamInsExperimentId.H2Mol_Q2_SU2_Max, ParamInsExperimentId.H2Mol_Q2_SU2_Min, ParamInsExperimentId.H2Mol_Q2_An_SU2_Min, ParamInsExperimentId.H2Mol_Q2_An_SU2_Max]
            if self.reps == 0:
                return 4
            else:
                return 5
    
    def get_ibm_ansatz_alg(self, any_noise_model: NoiseModel) -> AlgorithmNode:
        actions = self.get_ibm_ansatz_actions(any_noise_model)
        first_algorithm_node = None
        prev_algorithm_node = None
        
        for action in actions:
            current_algorithm_node = AlgorithmNode(action.name, action.instruction_sequence)
            if first_algorithm_node is None:
                assert prev_algorithm_node is None
                first_algorithm_node = current_algorithm_node
            else:
                assert prev_algorithm_node is not None
                assert isinstance(prev_algorithm_node, AlgorithmNode)
                prev_algorithm_node.next_ins = current_algorithm_node
            prev_algorithm_node = current_algorithm_node
        return first_algorithm_node
        
    def get_reward(self, hybrid_state) -> float:
        qs, _ = hybrid_state
        assert isinstance(qs, QuantumState)
        if self.experiment_id in [ParamInsExperimentId.H2Mol_Q2_An_SU2_Min, ParamInsExperimentId.H2Mol_Q2_An_SU2_Max]:
            rho = qs.single_partial_trace(index=self.embedding[1]) # get the density matrix of the partial trace that does not contains the ancilla
            energy = np_get_energy_from_rho(self.hamiltonian, rho)
        else:
            energy = np_get_energy(self.hamiltonian, qs.to_np_array())
        if self.experiment_id in [ParamInsExperimentId.H2Mol_Q1, ParamInsExperimentId.H2Mol_Q1_SU2_Min, ParamInsExperimentId.H2Mol_Q2_SU2_Min,ParamInsExperimentId.H2Mol_Q2_An_SU2_Min]:
            return energy
        else:
            assert self.experiment_id in [ParamInsExperimentId.H2Mol_Q1_SU2_Max, ParamInsExperimentId.H2Mol_Q2_SU2_Max,ParamInsExperimentId.H2Mol_Q2_An_SU2_Max]
            if isclose(energy, self.target_energy, abs_tol=Precision.isclose_abstol):
                    return 1.00
            else:
                return 0.00
        
    def ibm_get_reward(self, state: np.array) -> float:
        assert isinstance(state, np.array)

        qs = np_array_to_qs(state, list(self.embedding.values()))
        
        return self.get_reward((qs, None)) # the second component should be a classical state, but for these experiments does not matter
    
    
    def get_my_results(self, noise_model: NoiseModel, config, minimization_method: str):
        assert minimization_method in MINIMIZATION_METHODS
        actions = get_actions(noise_model, self.embedding, self.experiment_id, self.reps)
        energy_history = []
        result = minimize(cost_function, self.initial_parameters, 
                                args=(noise_model, actions, config, problem_instance, energy_history, project_settings, config_path), 
                                method=minimization_method)
        expected_energy = result.fun
        assert isclose(expected_energy, problem_instance.target_energy, rel_tol=Precision.rel_tol) or (expected_energy > problem_instance.target_energy)
        params_value = result.x
        n_iterations = result.nit # number of iterations
        if config["opt_technique"]:
            expected_energy = -expected_energy
        return params_value, expected_energy, n_iterations
    
    def get_ibm_results(self, minimization_method: str, basis_gates: BasisGates):
        assert minimization_method in MINIMIZATION_METHODS
        qc = QuantumCircuit(1)
        self.get_ibm_initial_state(qc, basis_gates) # reference state
        self.get_ibm_ansatz(qc) # appends ansatz for this experiment
        
        estimator = Estimator()
        result = minimize(cost_func_vqe, self.initial_parameters, args=(qc, self.hamiltonian, estimator), method="SLSQP")
        expected_energy = result.fun
        params_value = result.x
        n_iterations = result.nit
        return params_value, expected_energy, n_iterations
    
    
        
    def get_my_simulated_energy(self, config, noise_model: NoiseModel, params: List[float], reps, factor=1):
        assert config["min_horizon"] == config["max_horizon"]
        # fix this {0:0} line below
        parametric_actions = get_actions(noise_model, {0:0}, self.experiment_id, reps=reps)
        actions = get_binded_actions(parametric_actions, params, optimize=True)
        actions_to_instructions = dict()
        for action in actions:
            actions_to_instructions[action.name] = action.instruction_sequence
        actions_to_instructions["halt"] = []
        algorithms_path = os.path.join(config["output_dir"], "algorithms")
        algorithm_path = os.path.join(algorithms_path, f"{noise_model.hardware_spec.value}_0_{config['max_horizon']}.json")
        
        # load algorithm json
        f_algorithm = open(algorithm_path)
        alg = AlgorithmNode(serialized=json.load(f_algorithm), actions_to_instructions=actions_to_instructions)
        f_algorithm.close()  
        
        expected_energy = 0
        qr = QuantumRegister(len(self.embedding.keys()))
        cr = ClassicalRegister(len(self.embedding.keys()))
        qc = QuantumCircuit(qr, cr)
        
        self.get_ibm_initial_state(qc, noise_model.basis_gates)
        execute_algorithm(alg, qc, cbits=cr)
        qc.save_statevector('res', pershot=True)
        
        for key in self.embedding:
            initial_layout = {qr[key]: self.embedding[key]}
        ibm_noise_model = get_ibm_noise_model(noise_model.hardware_spec, thermal_relaxation=WITH_THERMALIZATION)
        
        # execute algorithm
        for _ in range(factor):
            state_vs = np.array(ibm_simulate_circuit(qc, ibm_noise_model, initial_layout))
        
            # compute expected energy
            for state in state_vs:
                expected_energy += self.ibm_get_reward(state)
            
        expected_energy/=(1024*factor)
        return expected_energy
    
    def get_ibm_ansatz_sim_energy(self,  hardware_spec: HardwareSpec, params: List[float], basis_gates: BasisGates, factor=1):
        expected_energy = 0
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        self.get_ibm_initial_state(qc, basis_gates)
        self.get_ibm_ansatz(qc)
        qc.save_statevector('res', pershot=True)
        
        # bind parameters
        params_dict = dict(zip(qc.parameters, params))
        qc = qc.bind_parameters(params_dict)
        
        initial_layout = {qr[0]: self.embedding[0]}
        ibm_noise_model = get_ibm_noise_model(hardware_spec, thermal_relaxation=WITH_THERMALIZATION)
        for _ in range(factor):
            state_vs = np.array(ibm_simulate_circuit(qc, ibm_noise_model, initial_layout))
            for state in state_vs:
                expected_energy += self.ibm_get_reward(self.hamiltonian, state)
        expected_energy/=(1024*factor)
        return expected_energy

def cost_func_vqe(params, ansatz, hamiltonian, estimator):
    """Return estimate of energy from estimator

    Parameters:
        params (ndarray): Array of ansatz parameters
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (Estimator): Estimator primitive instance

    Returns:
        float: Energy estimate
    """
    assert isinstance(estimator, Estimator)
    pub = (ansatz, hamiltonian, params)
    cost = estimator.run([ansatz], [hamiltonian], [params]).result().values[0]
    return cost
        
def get_actions(noise_model: NoiseModel, embedding: Dict[int,int], experiment_id: ParamInsExperimentId, reps) -> List[Action]:
   
    if experiment_id == ParamInsExperimentId.H2Mol_Q1:
        U3_gate = Instruction(embedding[0], Op.U3, params=['a', 'b', 'c'], symbols=['a', 'b', 'c']).to_basis_gate_impl(noise_model.basis_gates)
        return [POMDPAction("U3", U3_gate)]
    if experiment_id in [ParamInsExperimentId.H2Mol_Q1_SU2_Min, ParamInsExperimentId.H2Mol_Q1_SU2_Max]:
        # ansatz = EfficientSU2(2, su2_gates=["rx", "ry"], entanglement="linear", reps=1)
        # ansatz.decompose().draw("mpl") --> Rx1 - Ry1 - Rx2 - Ry2
        Rx1_gate = Instruction(embedding[0], Op.RX, params=['a'], symbols=['a']).to_basis_gate_impl(noise_model.basis_gates)
        Rx2_gate = Instruction(embedding[0], Op.RX, params=['b'], symbols=['b']).to_basis_gate_impl(noise_model.basis_gates)
        
        Ry1_gate = Instruction(embedding[0], Op.RY, params=['c'], symbols=['c'])
        Ry2_gate = Instruction(embedding[0], Op.RY, params=['d'], symbols=['d'])
        if reps == 0:
            return [POMDPAction("Rx1", Rx1_gate), POMDPAction("Ry1", Ry1_gate)]
        else:
            assert reps == 1
            return [POMDPAction("Rx1", Rx1_gate), POMDPAction("Rx2", Rx2_gate), POMDPAction("Ry1", Ry1_gate), POMDPAction("Ry2", Ry2_gate)]
    if experiment_id in [ParamInsExperimentId.H2Mol_Q2_SU2_Max, ParamInsExperimentId.H2Mol_Q2_SU2_Min]:
        Rx1_gate = Instruction(embedding[0], Op.RX, params=['a'], symbols=['a']).to_basis_gate_impl(noise_model.basis_gates)
        Ry1_gate = Instruction(embedding[0], Op.RY, params=['b'], symbols=['b']).to_basis_gate_impl(noise_model.basis_gates)
        
        
        Rx2_gate = Instruction(embedding[1], Op.RX, params=['c'], symbols=['c']).to_basis_gate_impl(noise_model.basis_gates)
        Ry2_gate = Instruction(embedding[1], Op.RY, params=['d'], symbols=['d']).to_basis_gate_impl(noise_model.basis_gates)
        
        if reps == 0:
            return [POMDPAction("Rx1", Rx1_gate), POMDPAction("Rx2", Rx2_gate),  POMDPAction("Ry1", Ry1_gate), POMDPAction("Ry2", Ry2_gate)]
        elif reps == 1:
            RxRy1 = Rx1_gate + Ry1_gate # first top layer
            RxRy2 = Rx2_gate + Ry2_gate # first bottom layer
            CX_gate = Instruction(embedding[1], Op.CNOT, control=embedding[0]).to_basis_gate_impl(noise_model.basis_gates)
            Rx3_gate = Instruction(embedding[0], Op.RX, params=['e'], symbols=['e']).to_basis_gate_impl(noise_model.basis_gates)
            Ry3_gate = Instruction(embedding[0], Op.RY, params=['f'], symbols=['f']).to_basis_gate_impl(noise_model.basis_gates)
            
            RxRy3 = Rx3_gate + Ry3_gate # second top layer
            
            Rx4_gate = Instruction(embedding[1], Op.RX, params=['g'], symbols=['g']).to_basis_gate_impl(noise_model.basis_gates)
            Ry4_gate = Instruction(embedding[1], Op.RY, params=['h'], symbols=['h']).to_basis_gate_impl(noise_model.basis_gates)
            RxRy4 = Rx4_gate + Ry4_gate # second bottom layer
            return [POMDPAction("RxRy1", RxRy1),  POMDPAction("RxRy2", RxRy2), POMDPAction("CX",CX_gate),POMDPAction("RxRy3", RxRy3),  POMDPAction("RxRy4", RxRy4)]
    if experiment_id in [ParamInsExperimentId.H2Mol_Q2_An_SU2_Min, ParamInsExperimentId.H2Mol_Q2_An_SU2_Max]:
        Rx1_gate = Instruction(embedding[0], Op.RX, params=['a'], symbols=['a']).to_basis_gate_impl(noise_model.basis_gates)
        Ry1_gate = Instruction(embedding[0], Op.RY, params=['b'], symbols=['b']).to_basis_gate_impl(noise_model.basis_gates)
        
        RxRy1 = Rx1_gate + Ry1_gate
        
        Rx2_gate = Instruction(embedding[0], Op.RX, params=['c'], symbols=['c']).to_basis_gate_impl(noise_model.basis_gates)
        Ry2_gate = Instruction(embedding[0], Op.RY, params=['d'], symbols=['d']).to_basis_gate_impl(noise_model.basis_gates)
        RxRy2 = Rx2_gate + Ry2_gate
        
        CX_gate = Instruction(embedding[1], Op.CNOT, control=embedding[0]).to_basis_gate_impl(noise_model.basis_gates)
        
        U3_gate = Instruction(embedding[1], Op.U3, params=['e', 'f', 'g'], symbols=['e', 'f', 'g']).to_basis_gate_impl(noise_model.basis_gates)
        
        MEAS_gate = Instruction(embedding[1], Op.MEAS)
        
        return [POMDPAction("RxRy1", RxRy1),  POMDPAction("RxRy2", RxRy2), POMDPAction("U3", U3_gate), POMDPAction("CX", CX_gate), POMDPAction("MEAS", [MEAS_gate])]
    else:
        raise Exception("Not implemented!")
    
def get_num_params_experiment(experiment_id: ParamInsExperimentId, reps) -> int:
    noise_model = NoiseModel(HardwareSpec.TENERIFE, False)
    embedding = {0:0, 1:1, 2:2} # any embedding, does not matter for this purpose
    actions = get_actions(noise_model, embedding, experiment_id, reps)
    parameters = set()
    for action in actions:
        assert isinstance(action, POMDPAction)
        for sym in action.symbols:
            parameters.add(sym)
    return len(parameters)

def get_hardware_embeddings(hardware: HardwareSpec, experiment_id, statistics=None, write=False) -> List[Dict[int, int]]:
    noise_model = NoiseModel(hardware, thermal_relaxation=WITH_THERMALIZATION)    
    answer = []
    pivot_qubits = set()
    if experiment_id in [ParamInsExperimentId.H2Mol_Q1]:
        # if "statistics" in kwargs.keys():
        #     kwargs["statistics"][hardware] = []
        if noise_model.basis_gates in [BasisGates.TYPE1, BasisGates.TYPE6]:
            for reverse in [False]:
                most_noisy_U3 = noise_model.get_most_noisy_qubit(Op.U3, reverse=reverse)[0]
                # if "statistics" in kwargs.keys():
                    # kwargs["statistics"][hardware].append((most_noisy_U3, Op.U3, not reverse))
                pivot_qubits.add(most_noisy_U3[1])
        else:
            for reverse in [False]:
                most_noisy_SX = noise_model.get_most_noisy_qubit(Op.SX, reverse=reverse)[0]
                # if "statistics" in kwargs.keys():
                #     kwargs["statistics"][hardware].append((most_noisy_SX, Op.SX, not reverse))
                pivot_qubits.add(most_noisy_SX[1])
        assert len(pivot_qubits) == 1
        for p in pivot_qubits:
            answer.append({0: p})
        return answer
    elif experiment_id in [ParamInsExperimentId.H2Mol_Q1_SU2_Max, ParamInsExperimentId.H2Mol_Q1_SU2_Min]:
        if noise_model.basis_gates in [BasisGates.TYPE1, BasisGates.TYPE6]:
            # we get the most noisy qubits in terms of U1 and U2
            for reverse in [False, True]:
                # U3 is used in Ry gate
                most_noisy_U3 = noise_model.get_most_noisy_qubit(Op.U3, reverse=reverse)[0]
                if reverse == False and most_noisy_U3[0] < 0.99:
                    pivot_qubits.add(most_noisy_U3[1])
                    if write:
                        statistics_file.write(f"{hardware.value},{most_noisy_U3[1]},{Op.U3.name},{round(most_noisy_U3[0],3)},{reverse}\n")
                
                # U1 is used in Rz gate
                most_noisy_U1 = noise_model.get_most_noisy_qubit(Op.U1, reverse=reverse)[0]
                
                if reverse == False and most_noisy_U1[0] < 0.99:
                    pivot_qubits.add(most_noisy_U1[1])
                    if write:
                        
                        statistics_file.write(f"{hardware.value},{most_noisy_U1[1]},{Op.U1.name},{round(most_noisy_U1[0],3)},{reverse}\n")
            
        else:
            assert noise_model.basis_gates in [BasisGates.TYPE2, BasisGates.TYPE3, BasisGates.TYPE7]
            for reverse in [False, True]:
                # for some reason, IBM does not has error models for RZ gates
                # we get the most noisy qubits in terms of SX (Ry gates uses them)
                most_noisy_SX = noise_model.get_most_noisy_qubit(Op.SX, reverse=reverse)[0]
                if reverse is False and most_noisy_SX[0] < 0.99:
                    pivot_qubits.add(most_noisy_SX[1])
                    if write:
                        statistics_file.write(f"{hardware.value},{most_noisy_SX[1]},{Op.SX.name},{round(most_noisy_SX[0],3)},{reverse}\n")
        
        for p in pivot_qubits:
            answer.append({0: p})
        return answer
    else:
        if experiment_id not in [ParamInsExperimentId.H2Mol_Q2_SU2_Min, ParamInsExperimentId.H2Mol_Q2_SU2_Max, ParamInsExperimentId.H2Mol_Q2_An_SU2_Min, ParamInsExperimentId.H2Mol_Q2_An_SU2_Max]:
            raise Exception(f"Experiment id {experiment_id} not valid")
        # Use most/least noise CX gate to choose two qubits
        
        for reverse in [False, True]:
            most_noisy_CX = noise_model.get_most_noisy_qubit(Op.CNOT, reverse=reverse)[0]
            target = most_noisy_CX[1][0]
            control = most_noisy_CX[1][1]
            if reverse == False and most_noisy_CX[0] < 0.99:
                answer.append({0: control, 1: target})
                if write:
                    statistics_file.write(f"{hardware.value},{most_noisy_CX[1]},{Op.CNOT.name},{round(most_noisy_CX[0],3)},{reverse}\n")

        return answer

def get_experiment_batches(experiment_id: ParamInsExperimentId,  reps=None):
    batches = dict()
    if experiment_id == ParamInsExperimentId.H2Mol_Q1:
        for hardware in P0_ALLOWED_HARDWARE:
            batches[hardware.value] = [hardware.value]
    else:
        assert reps is not None
        for hardware in HardwareSpec:
            embeddings = get_hardware_embeddings(hardware, experiment_id=experiment_id)
            for (index, _) in enumerate(embeddings):
                batches[f"{hardware.value}-{index}-{reps}"] = [hardware.value]
            
    return batches
        
        

def get_binded_actions(parametric_actions, params, optimize=False):
    # bind params
    bind_dict = dict()
    current_param_index = 0
    for parametric_action in parametric_actions:
        for symbol in parametric_action.symbols:
            if symbol not in bind_dict.keys():
                bind_dict[symbol] = params[current_param_index]
                current_param_index += 1
    assert current_param_index == len(params)
    actions = []
    for parametric_action in parametric_actions:
        actions.append(parametric_action.bind_symbols_from_dict(bind_dict))
        
    if optimize:
        for action in actions:
            action.optimize()
        
    return actions
        
def cost_function(params: List[float], noise_model: NoiseModel, parametric_actions: List[POMDPAction], config: Dict[Any, Any], problem_instance: ParamInsInstance, energy_history: List[float], project_settings: Dict[str, str], config_path: str):
    horizon = config["max_horizon"]
    
    hardware_str = config["hardware"][0]
    output_path = os.path.join(config["output_dir"], "pomdps", f"{hardware_str}_latest.txt")
    
    actions = get_binded_actions(parametric_actions, params, optimize=True)
    
    pomdp = build_pomdp(actions, noise_model, horizon, problem_instance.embedding, initial_state=problem_instance.initial_state)
    # pomdp.optimize_graph(problem_instance) # no optimization because every vertex has its own energy
    
    # save POMDP
    pomdp.serialize(problem_instance, output_path) 
    
    # now that pomdp is ready we must run bellman equation
    energy = get_bellman_value(project_settings, config_path)
    energy_history.append(energy)
    
    if config["opt_technique"] == "max":    
        return -energy
    else:
        return energy
    
    
def get_experiment_reps_values(experiment_id: ParamInsExperimentId) -> List[int]:
    reps_values = [0,1]
    return reps_values


if __name__ == "__main__":
    arg_backend = sys.argv[1]
    Precision.PRECISION = MAX_PRECISION
    Precision.update_threshold()
    
    if arg_backend == "gen_configs":
        for experiment_id in ParamInsExperimentId:
            if experiment_id != ParamInsExperimentId.H2Mol_Q1:
                reps_values = get_experiment_reps_values(experiment_id)
                if experiment_id == ParamInsExperimentId.H2Mol_Q1_SU2_Min:
                    horizons = [7, 5]
                    opt_technique = "min"
                if experiment_id == ParamInsExperimentId.H2Mol_Q1_SU2_Max:
                    horizons = [7, 5]
                    opt_technique = "max"
                if experiment_id == ParamInsExperimentId.H2Mol_Q2_An_SU2_Min:
                    horizons = [6]
                    opt_technique = "min"
                if experiment_id == ParamInsExperimentId.H2Mol_Q2_An_SU2_Max:
                    horizons = [6]
                    opt_technique = "max"
                if experiment_id == ParamInsExperimentId.H2Mol_Q2_SU2_Min:
                    horizons = [5, 5]
                    opt_technique = "min"
                if experiment_id == ParamInsExperimentId.H2Mol_Q2_SU2_Max:
                    horizons = [5, 5]
                    opt_technique = "max"
                assert len(reps_values) ==len(horizons)
                
                    
                for (horizon, reps) in zip(horizons, reps_values):
                    # experiment_id/hardware_spec-embeddingIndex-reps/{{results}}
                    batches = get_experiment_batches(experiment_id, reps=reps)
                    generate_configs("param_ins", experiment_id, min_horizon=horizon, max_horizon=horizon, allowed_hardware=HardwareSpec, batches=batches, opt_technique=opt_technique, reps=reps)
        
    elif arg_backend == "embeddings":
        settings = get_project_settings()
        for experiment_id in ParamInsExperimentId:
            if experiment_id != ParamInsExperimentId.H2Mol_Q1:    
                reps_values = get_experiment_reps_values(experiment_id)
                for reps in reps_values:
                    statistics_file = open(os.path.join(settings["PROJECT_PATH"], "results", "param_ins", experiment_id.value, "embeddings_stats.csv"), "w")
                    statistics_file.write("hardware_spec,address,instruction,succ_prob,least_noisy\n")
                    batches = get_experiment_batches(experiment_id, reps=reps)
                    for batch_name in batches.keys():
                        config_path = get_config_path("param_ins", experiment_id, batch_name)
                        config = load_config_file(config_path, ParamInsExperimentId)
                        directory_exists(config["output_dir"])
                        assert len(config["hardware"]) == 1
                        hardware_spec = find_enum_object(config["hardware"][0], HardwareSpec)
                        embedding_index = int(batch_name.split("-")[1])
                        if embedding_index == 0:
                            embeddings = get_hardware_embeddings(hardware_spec, experiment_id=experiment_id, statistics=statistics_file, write=True)
                        else:
                            embeddings = get_hardware_embeddings(hardware_spec, experiment_id=experiment_id, write=False)
                        generate_embeddings_files(config, hardware_spec, batch_name, embeddings)
                    statistics_file.close()
    elif arg_backend == "exp_q1": # runs ParamInsExperimentId.H2Mol_Q1
        project_settings = get_project_settings()
        experiment_id = ParamInsExperimentId.H2Mol_Q1
        batches = get_experiment_batches(experiment_id)
        for batch_name in batches.keys():
            config_path = get_config_path("param_ins", experiment_id, batch_name)
            config = load_config_file(config_path, ParamInsExperimentId)
            directory_exists(config["output_dir"])
            
            results_summary_path = os.path.join(config["output_dir"], "summary_results.csv")
            results_summary_file = open(results_summary_path, "w")
            line = "my_energy,ibm_energy,my_real_energy,ibm_real_energy,my_nit,ibm_nit,minimizer,my_theta,ibm_theta,my_phi,ibm_phi,my_lambda,ibm_lambda\n"
            results_summary_file.write(line)
            
            # generate embeddings
            generate_embeddings(config_path=config_path, experiment_enum=ParamInsExperimentId, experiment_id=ParamInsExperimentId.H2Mol_Q1, get_hardware_embeddings=get_hardware_embeddings)
            
            output_folder = os.path.join(config["output_dir"], "pomdps")
            # check that there is a folder with the experiment id inside pomdps path
            directory_exists(output_folder)
            
            assert len(config["hardware"]) == 1
            hardware_spec = find_enum_object(config["hardware"][0], HardwareSpec)
            noise_model = NoiseModel(hardware_spec, thermal_relaxation=WITH_THERMALIZATION)
            
            embeddings = default_load_embeddings(config, ParamInsExperimentId)
            assert embeddings["count"] == 1
            embeddings = embeddings[hardware_spec]["embeddings"]
            assert len(embeddings) == 1
            embedding = embeddings[0]
            
            for minimization_method in MINIMIZATION_METHODS:
                problem_instance = ParamInsInstance(embedding, ParamInsExperimentId.H2Mol_Q1)
                
                # getting my methods results
                my_params_value, my_energy, my_n_iterations = problem_instance.get_my_results(noise_model, config, minimization_method)
                assert len(my_params_value) == 3
                my_theta = round(my_params_value[0],3)
                my_phi = round(my_params_value[1],3)
                my_lambda = round(my_params_value[2],3)
            
                
                # getting ibm method results
                ibm_params_value, ibm_energy, ibm_n_iterations = problem_instance.get_ibm_results(minimization_method, noise_model.basis_gates)
                assert len(ibm_params_value) == 3
                ibm_theta = round(ibm_params_value[0],3)
                ibm_phi = round(ibm_params_value[1],3)
                ibm_lambda = round(ibm_params_value[2],3)
                
                # results of simulator with the circuit that we synthesize
                my_real_energy = problem_instance.get_my_simulated_energy(noise_model, config, my_params_value, reps=0)
                
                # results of simulator with the binded ansatz that ibm finds
                ibm_real_energy = problem_instance.get_ibm_ansatz_sim_energy(hardware_spec, ibm_params_value, noise_model.basis_gates)
                
                line = f"{my_energy},{ibm_energy},{my_real_energy},{ibm_real_energy},{my_n_iterations},{ibm_n_iterations},{minimization_method},{my_theta},{ibm_theta},{my_phi},{ibm_phi},{my_lambda},{ibm_lambda}\n"
                print(line)
                results_summary_file.write(line)
                
            results_summary_file.close()
    elif arg_backend[1] == "run_config":        
        if arg_backend == "Q1SU2Min":
            experiment_id = ParamInsExperimentId.H2Mol_Q1_SU2_Min
        if arg_backend == "Q1SU2Max":
            experiment_id = ParamInsExperimentId.H2Mol_Q1_SU2_Max
        if arg_backend == "Q2AnMin":
            experiment_id = ParamInsExperimentId.H2Mol_Q2_An_SU2_Min
        if arg_backend == "Q2AnMax":
            experiment_id = ParamInsExperimentId.H2Mol_Q2_An_SU2_Max
        if arg_backend == "Q2Min":
            experiment_id = ParamInsExperimentId.H2Mol_Q2_SU2_Min
        if arg_backend == "Q2Max":
            experiment_id = ParamInsExperimentId.H2Mol_Q2_SU2_Max
        project_settings = get_project_settings()
        batches = get_experiment_batches(experiment_id)
        for batch_name in batches.keys():
            config_path = get_config_path("param_ins", experiment_id, batch_name)
            config = load_config_file(config_path, ParamInsExperimentId)
            
            # retrieving hardware spec
            assert len(config["hardware"]) == 1
            hardware_spec = find_enum_object(config["hardware"][0], HardwareSpec)
            
            noise_model = NoiseModel(hardware_spec, WITH_THERMALIZATION)
            
            # load embeddings
            embeddings = default_load_embeddings(config, ParamInsExperimentId)
            assert embeddings["count"] == 1
            embeddings = embeddings[hardware_spec]["embeddings"]
            assert len(embeddings) == 1
            embedding = embeddings[0]
            
            # settings up file that stores results
            all_results_path = os.path.join(config_path["output_dir"], "results.csv")
            all_results_file = open(all_results_path, "w")
            
            reps = int(batch_name.split("-")[2])
            
            opt_technique = config["opt_technique"]
            if opt_technique  == "min":
                computed_value = "energy"
            else:
                computed_value = "prob"
                
            # writting header
            header_elements = [
                f"my_{computed_value}", f"ibm_{computed_value}",
                f"sim_my_{computed_value}", f"sim_my_{computed_value}",
                f"mc_my_{computed_value}", f"mc_ibm_{computed_value}",
                "my_nit", "ibm_nit", "reps" ,"opt_technique", "minimization_method",
                "my_time","ibm_time"
            ]
            
            num_params = get_num_params_experiment(experiment_id, reps)
            
            for index in range(num_params):
                header_elements.append(f"my_p{index}")
                header_elements.append(f"ibm_p{index}")
            all_results_file.write(",".join(header_elements) + "\n")
            
            for minimization_method in MINIMIZATION_METHODS:
                problem_instance = ParamInsInstance(embedding, experiment_id, reps=reps)
                
                # we run a cost function that corresponds to Bellman equation
                start_time = time.time()
                my_params, my_computed_value, my_nit = problem_instance.get_my_results(noise_model, config, minimization_method)
                end_time = time.time()
                my_time = round(end_time-start_time, 3)
                
                # run ibm variational algorithm
                start_time = time.time()
                ibm_params, ibm_computed_value, ibm_nit = problem_instance.get_ibm_results(minimization_method, noise_model.basis_gates)
                end_time = time.time()
                ibm_time = round(end_time - start_time, 3)
                
                # simulation results
                sim_my_computed_value = problem_instance.get_my_simulated_energy(noise_model, my_params, reps)
                sim_ibm_computed_value = problem_instance.get_ibm_simulated_energy(hardware_spec, ibm_params, noise_model.basis_gates, factor=2)
                
                # results from Markov Chain (we already know the algorithm and resolve the choices)
                algorithm_path = os.path.join(config["output_dir"], "algorithms", f"{hardware_spec.value}_0_{config['max_horizon']}.json")
                pomdp_path = os.path.join(config["output_dir"], "pomdps", f"{hardware_spec.value}_latest.txt")
                mc_my_computed_value = get_markov_chain_results(project_settings, algorithm_path, pomdp_path)
                
                parametric_actions = problem_instance.get_ibm_ansatz_actions(noise_model)
                actions = get_binded_actions(parametric_actions, ibm_params)
                ibm_pomdp = build_pomdp(actions, noise_model, problem_instance.get_ibm_horizon(), problem_instance.embedding, initial_state=problem_instance.initial_state)
                ibm_pomdp_path = os.path.join(config["output_dir"], "pomdps", f"ibm_{hardware_spec.value}_latest.txt")
                ibm_pomdp.serialize(problem_instance, ibm_pomdp_path)
                
                ibm_algorithm_path = os.path.join(config["output_dir"], "algorithms", f"ibm_{hardware_spec.value}.json")
                ibm_algorithm_node = problem_instance.get_ibm_ansatz_alg()
                assert isinstance(ibm_algorithm_node, AlgorithmNode)
                algorithm_file = open(ibm_algorithm_path, "w")
                algorithm_file.write(json.dump(ibm_algorithm_node.serialize()))
                mc_ibm_computed_value = get_markov_chain_results(project_settings, ibm_algorithm_path, ibm_pomdp_path)
                
                line_elements = [
                   my_computed_value, ibm_computed_value,
                    sim_my_computed_value, sim_ibm_computed_value,
                    mc_my_computed_value, mc_ibm_computed_value,
                    my_nit, ibm_nit, reps, opt_technique, minimization_method,
                    my_time,ibm_time
                ]
                
                assert len(my_params) == len(ibm_params)
                
                for index in range(len(my_params)):
                    line_elements.append(str(round(my_params[index],3)))
                    line_elements.append(str(round(ibm_params[index],3)))
                all_results_file.write(",".join(line_elements) + "\n")
        all_results_file.close()
    else:
        
        raise Exception("first argument does not match anything")
        
        
        
       