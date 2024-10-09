from argparse import Action
from enum import Enum
import os, sys

from ibm_noise_models import NoiseModel
sys.path.append(os.getcwd() + "/..")
from typing import Dict, List

from cmemory import ClassicalState
from qstates import QuantumState
from utils import Precision
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature import settings

settings.use_pauli_sum_op = False

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter, ParityMapper
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

MAX_PRECISION = 10

class H2InstructionSet(Enum):
    CliffordT = "Clifford+T" # tries to find unitary
    CliffordTAncilla = "Clifford+T_w_ancilla" # see if using an ancilla helps
    
class H2Problem(Enum):
    P0 = "P1" # This consists of finding the ground state of an hydrogen molecule using Z2Symmetries
    P1 = "P1" # https://static-content.springer.com/esm/art%3A10.1038%2Fs41534-019-0187-2/MediaObjects/41534_2019_187_MOESM1_ESM.pdf
    P2 = "P2" # https://learning.quantum.ibm.com/course/variational-algorithm-design/examples-and-applications#quantum-chemistry-ground-state-and-excited-energy-solver


def get_hamiltonian(problem: H2Problem) -> SparsePauliOp:
    if problem == H2Problem.P0:
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
        return qubit_op
    else:
        raise Exception(f"hamiltonian for {problem} not implemented!")

class H2MoleculeInstance:
    def __init__(self, embedding: Dict[int, int]) -> None:
        """_summary_

        Args:
            embedding (Dict[int, int]): mapping from logical qubits to physical qubits
        """        
        self.embedding = embedding
        self.num_qubits = max()
        self.initial_state = None # must be a hybrid quantum state
        self.get_initial_states()
        
    def get_initial_states(self):
        self.initial_state = (QuantumState(0), ClassicalState())
        
    def is_target_qs(self, hybrid_state) -> bool:
        pass
    
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
            
def get_actions(noise_model: NoiseModel, embedding: Dict[int,int], instruction_set: H2InstructionSet) -> List[Action]:
    if instruction_set == H2InstructionSet.CliffordT:
        pass
    else:
        assert instruction_set == H2InstructionSet.CliffordTAncilla

if __name__ == "__main__":
    arg_backend = sys.argv[1]
    Precision.PRECISION = MAX_PRECISION
    Precision.update_threshold()
    
    if arg_backend == "test":
        Test.test_hamiltonian()
    
    
    