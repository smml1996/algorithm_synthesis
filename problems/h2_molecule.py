from argparse import Action
from cmath import isclose, pi
from enum import Enum
import time
import numpy as np
import json
import os, sys
from scipy.linalg import expm
sys.path.append(os.getcwd() + "/..")

from experiments_utils import default_load_embeddings, directory_exists, generate_configs, generate_embeddings, get_config_path, get_embeddings_path, get_project_settings
from pomdp import POMDPAction, build_pomdp, default_guard
import qmemory
from qpu_utils import BasisGates, Op


from ibm_noise_models import HardwareSpec, Instruction, NoiseModel, get_num_qubits_to_hardware, load_config_file
from typing import Dict, List
from utils import Precision, np_get_ground_state
from cmemory import ClassicalState
from qstates import QuantumState
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature import settings

settings.use_pauli_sum_op = False

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter, ParityMapper
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

MAX_PRECISION = 10
WITH_THERMALIZATION = False
P0_ALLOWED_HARDWARE = [HardwareSpec.AUCKLAND, HardwareSpec.WASHINGTON, HardwareSpec.ROCHESTER]
    
class H2ExperimentID(Enum):
    P0_CliffordT = "P0_CliffordT" # This consists of finding the ground state of an hydrogen molecule using Z2Symmetries -- "Clifford+T" instruction set
    P0_Rotation = "P0_Rotation" # using Rx(pi/2), Rz(pi/2), Ry(pi/2)
    # P1 = "P1" # https://static-content.springer.com/esm/art%3A10.1038%2Fs41534-019-0187-2/MediaObjects/41534_2019_187_MOESM1_ESM.pdf
    # P2 = "P2" # https://learning.quantum.ibm.com/course/variational-algorithm-design/examples-and-applications#quantum-chemistry-ground-state-and-excited-energy-solver


def get_hamiltonian(problem: H2ExperimentID) -> SparsePauliOp:
    if problem in [H2ExperimentID.P0_CliffordT, H2ExperimentID.P0_Rotation]:
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
    else:
        raise Exception(f"hamiltonian for {problem} not implemented!")
    
def schroedinger_equation(H: SparsePauliOp, t: complex, initial_state: np.array) -> np.array:
    I = complex(0, 1)

    # Compute the time evolution operator U = e^(iHt)
    U = expm(-I * H * t) # planck constant is assumed to be 1

    # Apply U to the initial state |q>
    final_state = np.dot(U, initial_state)
    
    # since U might not be a non-unitary matrix (for imaginary time evolution)
    final_state = normalize_np_array(final_state)
    return final_state
    
def normalize_np_array(quantum_state):
    # normalize quantum state
    norm = np.dot(np.conjugate(quantum_state).T, quantum_state)
    assert isclose(norm.imag, 0.0, abs_tol=Precision.isclose_abstol)
    norm = norm.real
    assert norm > 0
    sq_norm = np.sqrt(norm)
    return quantum_state/sq_norm
    
def get_energy(H: SparsePauliOp, quantum_state: np.array) -> float:
    quantum_state = normalize_np_array(quantum_state)
    # Compute the expectation value ⟨q|H|q⟩
    q_dagger = np.conjugate(quantum_state).T  # Conjugate transpose of |q>
    H_q = np.dot(H, quantum_state)            # Matrix multiplication H|q>
    expectation_value = np.dot(q_dagger, H_q)  # Inner product ⟨q|H|q⟩
    assert isclose(expectation_value.imag, 0.0, abs_tol=Precision.isclose_abstol)
    return expectation_value.real

def get_fidelity(state1: np.array, state2: np.array) -> float:
    result = np.dot(np.conjugate(state1).T, state2 )
    return result.real*np.conjugate(result.real)
    

class H2MoleculeInstance:
    def __init__(self, embedding, experiment_id: H2ExperimentID, t: int, is_imaginary=True) -> None:
        """_summary_

        Args:
            embedding (Dict[int, int]): mapping from logical qubits to physical qubits
        """        
        self.embedding = embedding
        self.experiment_id = experiment_id
        self.initial_state = None # must be a hybrid quantum state
        self.t = t
        self.H = get_hamiltonian(self.experiment_id)
        self.get_initial_states()
        
        # compute target state
        qs, _ = self.initial_state
        assert isinstance(qs, QuantumState)
        if is_imaginary:
            self.target_state = schroedinger_equation(get_hamiltonian(self.experiment_id), complex(0, -self.t), qs.to_np_array())
        else:
            self.target_state = schroedinger_equation(get_hamiltonian(self.experiment_id), self.t, qs.to_np_array())
        self.target_energy = get_energy(self.H, self.target_state)
        
    def get_initial_states(self):
        """the initial state must have non zero overlap with the ground state
        Raises:
            Exception: _description_
        """        
        classical_state = ClassicalState()
        quantum_state = None
        if self.experiment_id in [H2ExperimentID.P0_CliffordT, H2ExperimentID.P0_Rotation]:
            quantum_state = QuantumState(0, qubits_used=list(self.embedding.values()))
            X0 = Instruction(self.embedding[0], Op.X).get_gate_data()
            quantum_state = qmemory.handle_write(quantum_state, X0)
        else:
            raise Exception("Not implemented")
        assert quantum_state is not None
        self.initial_state = (quantum_state, classical_state)
        
    def is_target_qs(self, hybrid_state) -> bool:
        qs, cs = hybrid_state
        assert isinstance(qs, QuantumState)
        if self.experiment_id in [H2ExperimentID.P0_CliffordT, H2ExperimentID.P0_Rotation]:
            state = qs.to_np_array()
            energy = get_energy(self.H, state)
            return isclose(energy, self.target_energy, abs_tol=1/1e3)
    
def get_actions(noise_model: NoiseModel, embedding: Dict[int,int], experiment_id: H2ExperimentID) -> List[Action]:
   
    if experiment_id == H2ExperimentID.P0_CliffordT:
        t_gate = Instruction(embedding[0], Op.T).to_basis_gate_impl(noise_model.basis_gates)
        h_gate = Instruction(embedding[0], Op.H).to_basis_gate_impl(noise_model.basis_gates) 
        s_gate = Instruction(embedding[0], Op.S).to_basis_gate_impl(noise_model.basis_gates)
        return [POMDPAction("T0", t_gate), POMDPAction("H0", h_gate), POMDPAction("S0", s_gate)]
        
    elif experiment_id == H2ExperimentID.P0_Rotation:
        rx_gate = Instruction(embedding[0], Op.RX, params=[pi/2]).to_basis_gate_impl(noise_model.basis_gates)
        ry_gate = Instruction(embedding[0], Op.RY, params=[pi/2]).to_basis_gate_impl(noise_model.basis_gates)
        rz_gate = Instruction(embedding[0], Op.RZ, params=[pi/2]).to_basis_gate_impl(noise_model.basis_gates)
        return [POMDPAction("RZ0", rz_gate), POMDPAction("RX0", rx_gate), POMDPAction("RY0", ry_gate)]
    else:
        raise Exception("Not implemented!")

def get_hardware_embeddings(hardware: HardwareSpec, **kwargs) -> List[Dict[int, int]]:
    noise_model = NoiseModel(hardware, thermal_relaxation=WITH_THERMALIZATION)
    # assert hardware not in kwargs["statistics"].keys()
    kwargs["statistics"][hardware] = []
    if kwargs["experiment_id"] in [H2ExperimentID.P0_CliffordT, H2ExperimentID.P0_Rotation]:
        answer = []
        pivot_qubits = set()

        # get qubit with highest accumulated measurement error rate
        for reverse in [False, True]:
            most_noisy_meas = noise_model.get_most_noisy_qubit(Op.MEAS, reverse=reverse)[0]
            kwargs["statistics"][hardware].append((most_noisy_meas, Op.MEAS, not reverse))
            pivot_qubits.add(most_noisy_meas[1])
        if noise_model.basis_gates in [BasisGates.TYPE1, BasisGates.TYPE6]:
            # we get the most noisy qubits in terms of U1 and U2
            for reverse in [False, True]:
                most_noisy_U1 = noise_model.get_most_noisy_qubit(Op.U1, reverse=reverse)[0]
                most_noisy_U2 = noise_model.get_most_noisy_qubit(Op.U2, reverse=reverse)[0]
                kwargs["statistics"][hardware].append((most_noisy_U1, Op.U1, not reverse))
                kwargs["statistics"][hardware].append((most_noisy_U2, Op.U2, not reverse))
                pivot_qubits.add(most_noisy_U1[1])
                pivot_qubits.add(most_noisy_U2[1])
            
        else:
            for reverse in [False, True]:
                # for some reason, IBM does not has error models for RZ gates
                # we get the most noisy qubits in terms of SX and RZ gates
                most_noisy_SX = noise_model.get_most_noisy_qubit(Op.SX, reverse=reverse)[0]
                # most_noisy_RZ = noise_model.get_most_noisy_qubit(Op.RZ)[0]
                kwargs["statistics"][hardware].append((most_noisy_SX, Op.SX, not reverse))
                # kwargs["statistics"][hardware].append((most_noisy_RZ, Op.RZ))
                pivot_qubits.add(most_noisy_SX[1])
                # pivot_qubits.add(most_noisy_RZ[1])
            
        for p in pivot_qubits:
            answer.append({0: p})
        return answer
    else:
        raise Exception("Not implemented!")

class Test:
    @staticmethod
    def test_hamiltonian():
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
        # converter = QubitConverter(ParityMapper(), two_qubit_reduction=True)

        # qubit_op = converter.convert(hamiltonian, num_particles=es_problem.num_particles)
        mapper = JordanWignerMapper()
        converter = QubitConverter(mapper, z2symmetry_reduction="auto")
        
        qubit_op = converter.convert(
            hamiltonian,
            num_particles=problem.num_particles,
            sector_locator=problem.symmetry_sector_locator,
        )

        assert isinstance(qubit_op, SparsePauliOp)
        for pauli, coeff in sorted(qubit_op.label_iter()):
            print(f"{coeff.real:+.8f} * {pauli}")
    
    @staticmethod
    def check_ground(matrix, my_state, their_state, expected_energy=None):
        my_energy = get_energy(matrix, my_state)
        their_energy = get_energy(matrix, their_state)
        if expected_energy is not None:
            assert isclose(my_energy, expected_energy, rel_tol=Precision.rel_tol)
        if not isclose(my_energy, their_energy, rel_tol=Precision.rel_tol):
            raise Exception(f"Energies do not match: {my_energy} and {their_energy}")
        fidelity = get_fidelity(my_state, their_state)
        if not isclose(fidelity, 1.0, rel_tol=Precision.rel_tol):
            raise Exception(f"Fidelity is not close:\n {my_state}\n {their_state}")
        
    @staticmethod
    def pauli_test():
        # pauli matrices
        I = np.array([[1, 0], [0, 1]])

        X = np.array([[0, 1], [1, 0]])

        Z = np.array([[1,0], [0,-1]])

        Y = np.array([[0, complex(0, -1)], [complex(0, 1), 0]])
                
        # testing identity matrix
        my_ground_state = schroedinger_equation(I, complex(0,-14), np.array([1,0]))
        Test.check_ground(I, my_ground_state, [1, 0], expected_energy=1)
        
        # testing pauli-X
        eigenvalues, eigenvectors = np.linalg.eigh(X)
        their_ground_energy, eigenstates = np_get_ground_state(eigenvalues, eigenvectors)
        assert isclose(their_ground_energy, -1, rel_tol=Precision.rel_tol)
        assert len(eigenstates) == 1
        my_ground_state = schroedinger_equation(X, complex(0,-14), np.array([1,0]))
        Test.check_ground(X, my_ground_state, eigenstates[0], expected_energy=-1)
        
        # testing pauli-Z
        my_ground_state = schroedinger_equation(Z, complex(0,-14), np.array([1,0]))
        Test.check_ground(Z, my_ground_state, [1,0], expected_energy=1)
        
        my_ground_state = schroedinger_equation(Z, complex(0,-14), np.array([0,1]))
        Test.check_ground(Z, my_ground_state, [0,1], expected_energy=-1)
        
        my_ground_state = schroedinger_equation(Z, complex(0,-14), np.array([1/np.sqrt(2),1/np.sqrt(2)]))
        Test.check_ground(Z, my_ground_state, [0,1], expected_energy=-1)
        
        # testing pauli-Y
        eigenvalues, eigenvectors = np.linalg.eig(Y)
        their_ground_energy, eigenstates = np_get_ground_state(eigenvalues, eigenvectors)
        assert isclose(their_ground_energy, -1, rel_tol=Precision.rel_tol)
        assert len(eigenstates) == 1
        my_ground_state = schroedinger_equation(Y, complex(0,-14), np.array([1,0]))
        Test.check_ground(Y, my_ground_state, eigenstates[0], expected_energy=-1)
        
    @staticmethod
    def test_1Q_implementations(real_instruction: Instruction, implementations: List[List[Instruction]]):
        embedding = {0:0}
        q0 = QuantumState(0, qubits_used=[0])
        q1 = QuantumState(1, qubits_used=[0])
        basis_states = [q0, q1]
        
        target_state0 = qmemory.handle_write(q0, real_instruction.get_gate_data())
        target_state1 = qmemory.handle_write(q1, real_instruction.get_gate_data())
        target_states = [target_state0, target_state1]
        
        for (basis_state, target_state) in zip(basis_states, target_states):
            for (index, implementation) in enumerate(implementations):
                current_state = basis_state
                for instruction in implementation:
                    current_state = qmemory.handle_write(current_state, instruction.get_gate_data())
                if current_state != target_state:
                    raise Exception(f"Mismatch with implementation {index}.\n {target_state}\n {current_state}\n basis_state={basis_state}\nfidelity={get_fidelity(current_state.to_np_array(), target_state.to_np_array())}")
    
    @staticmethod
    def test_used_1Q_gates():
        test_instructions = [
            Instruction(0, Op.T),
            Instruction(0, Op.H),
            Instruction(0, Op.S),
            Instruction(0, Op.RX, params=[pi/2]),
            Instruction(0, Op.RY, params=[pi/2]),
            Instruction(0, Op.RZ, params=[pi/2])
        ] 
       
        for instruction in test_instructions:
            implementation1 = instruction.to_basis_gate_impl(BasisGates.TYPE1)
            implementation2 = instruction.to_basis_gate_impl(BasisGates.TYPE2)
            assert implementation1 != implementation2
            Test.test_1Q_implementations(instruction, [implementation1, implementation2])
        
       
        
                
    
            
def generate_pomdp(experiment_id, hardware_spec: HardwareSpec, 
                embedding: Dict[int, int], pomdp_write_path: str, t: int, max_horizon:int, return_pomdp=False):
    noise_model = NoiseModel(hardware_spec, thermal_relaxation=WITH_THERMALIZATION)
    h2_molecule_instance = H2MoleculeInstance(embedding, experiment_id, t)
    actions = get_actions(noise_model, embedding, experiment_id)
    initial_distribution = []
    initial_distribution.append((h2_molecule_instance.initial_state, 1.00))

    start_time = time.time()
    pomdp = build_pomdp(actions, noise_model, max_horizon, embedding, initial_distribution=initial_distribution)
    pomdp.optimize_graph(h2_molecule_instance)
    end_time = time.time()
    if return_pomdp:
        return pomdp
    pomdp.serialize(h2_molecule_instance, pomdp_write_path)
    return end_time-start_time
    
def generate_pomdps(config_path, t:int):
    assert isinstance(t, int)
    config = load_config_file(config_path, H2ExperimentID)
    experiment_id = config["experiment_id"]
    assert isinstance(experiment_id, H2ExperimentID)
    
    # the file that contains the time to generate the POMDP is in this folder
    directory_exists(config["output_dir"])
        
     # all pomdps will be outputed in this folder:
    output_folder = os.path.join(config["output_dir"], "pomdps")
    # check that there is a folder with the experiment id inside pomdps path
    directory_exists(output_folder)

    all_embeddings = default_load_embeddings(config, H2ExperimentID)
    
    times_file_path = os.path.join(config["output_dir"], 'pomdp_times.csv')
    times_file = open(times_file_path, "w")
    times_file.write("backend,embedding,time\n")
    for backend in HardwareSpec:
        if backend.value in config["hardware"]:
            # try:
            embeddings = all_embeddings[backend]["embeddings"]
            for (index, m) in enumerate(embeddings):
                print(backend, index, m)
                time_taken = generate_pomdp(experiment_id, backend, m, f"{output_folder}/{backend.value}_{index}.txt", t, config["max_horizon"])
                if time_taken is not None:
                    times_file.write(f"{backend.name},{index},{time_taken}\n")
                times_file.flush()
            # except Exception as err:
            #     print(f"Unexpected {err=}, {type(err)=}")
    times_file.close()
    


if __name__ == "__main__":
    arg_backend = sys.argv[1]
    Precision.PRECISION = MAX_PRECISION
    Precision.update_threshold()
    
    if arg_backend == "gen_configs":
        generate_configs("H2", H2ExperimentID.P0_CliffordT, 4, 10, allowed_hardware=P0_ALLOWED_HARDWARE)
        
        generate_configs("H2", H2ExperimentID.P0_Rotation, 4, 10, allowed_hardware=P0_ALLOWED_HARDWARE)
    if arg_backend == "embeddings":
        batches = get_num_qubits_to_hardware(WITH_THERMALIZATION, allowed_hardware=P0_ALLOWED_HARDWARE)
        statistics = dict()
        for num_qubits in batches.keys():
            for experiment_id in H2ExperimentID:
                config_path = get_config_path("H2", experiment_id, num_qubits)
                print(config_path)
                generate_embeddings(config_path=config_path, experiment_enum=H2ExperimentID, experiment_id=experiment_id, get_hardware_embeddings=get_hardware_embeddings, statistics=statistics)

    if arg_backend == "all_pomdps":
        batches = get_num_qubits_to_hardware(WITH_THERMALIZATION, allowed_hardware=P0_ALLOWED_HARDWARE)
        
        for num_qubits in batches.keys():
            for experiment_id in H2ExperimentID:
                config_path = get_config_path("H2", experiment_id, num_qubits)
                generate_pomdps(config_path, 14)
            
    if arg_backend == "test":
        # Test.test_hamiltonian()
        # Test.pauli_test()
        Test.test_used_1Q_gates()
    
    
    