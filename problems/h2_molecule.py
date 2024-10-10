from argparse import Action
from cmath import pi
from enum import Enum
import json
import os, sys
sys.path.append(os.getcwd() + "/..")

from experiments_utils import directory_exists, generate_configs, generate_embeddings, get_config_path, get_embeddings_path
from pomdp import POMDPAction
import qmemory
from qpu_utils import BasisGates, Op


from ibm_noise_models import HardwareSpec, Instruction, NoiseModel, get_num_qubits_to_hardware
from typing import Dict, List
from utils import Precision
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
    
class H2ExperimentID(Enum):
    P0_CliffordT = "P0_CliffordT" # This consists of finding the ground state of an hydrogen molecule using Z2Symmetries -- "Clifford+T" instruction set
    P1 = "P1" # https://static-content.springer.com/esm/art%3A10.1038%2Fs41534-019-0187-2/MediaObjects/41534_2019_187_MOESM1_ESM.pdf
    P2 = "P2" # https://learning.quantum.ibm.com/course/variational-algorithm-design/examples-and-applications#quantum-chemistry-ground-state-and-excited-energy-solver


def get_hamiltonian(problem: H2ExperimentID) -> SparsePauliOp:
    if problem == H2ExperimentID.P0:
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
    
def get_ground_states(hamiltonian: SparsePauliOp) -> List[QuantumState]:
    pass

class H2MoleculeInstance:
    def __init__(self, embedding: Dict[int, int], experiment_id: H2ExperimentID) -> None:
        """_summary_

        Args:
            embedding (Dict[int, int]): mapping from logical qubits to physical qubits
        """        
        self.embedding = embedding
        self.experiment_id = experiment_id
        self.initial_state = None # must be a hybrid quantum state
        self.get_initial_states()
        
    def get_initial_states(self):
        """the initial state must have non zero overlap with the ground state
        Raises:
            Exception: _description_
        """        
        classical_state = ClassicalState()
        quantum_state = None
        if self.experiment_id == H2ExperimentID.P0_CliffordT:
            quantum_state = QuantumState(0, qubits_used=list(self.embedding.values()))
            H0 = Instruction(self.embedding[0], Op.H).get_gate_data()
            
            quantum_state = qmemory.handle_write(quantum_state, H0)
        else:
            raise Exception("Not implemented")
        assert quantum_state is not None
        self.initial_state = (quantum_state, classical_state)
        
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
            
def get_actions(noise_model: NoiseModel, embedding: Dict[int,int], experiment_id: H2ExperimentID) -> List[Action]:
    if experiment_id == H2ExperimentID.P0_CliffordT:
        if noise_model.basis_gates in [BasisGates.TYPE1, BasisGates.TYPE6]:
            H0 = POMDPAction("H0", [Instruction(embedding[0], Op.U2, params=[0.0, pi])])
            T0 = POMDPAction("T0", [Instruction(embedding[0], Op.U1, params=[pi/4])])
            T0D = POMDPAction("T0D", [Instruction(embedding[0], Op.U1, params=[-pi/4])])
        else:
            assert noise_model.basis_gates in [BasisGates.TYPE2, BasisGates.TYPE3, BasisGates.TYPE4, BasisGates.TYPE7]
            H0 = POMDPAction("H0", [
                Instruction(embedding[0], Op.RZ, params=[pi/2]),
                Instruction(embedding[0], Op.SX),
                Instruction(embedding[0], Op.RZ, params=[pi/2])
            ])
            T0 = POMDPAction("T0", [Instruction(embedding[0], Op.RZ, params=[pi/4])])
            T0D = POMDPAction("T0D", [Instruction(embedding[0], Op.RZ, params=[-pi/4])])
        MEAS0 = POMDPAction("P0", [Instruction(embedding[0], Op.MEAS)])
        return [H0, T0, T0D, MEAS0]
    else:
        raise Exception("Not implemented!")




def get_hardware_embeddings(hardware: HardwareSpec, **kwargs) -> List[Dict[int, int]]:
    noise_model = NoiseModel(hardware, thermal_relaxation=WITH_THERMALIZATION)
    if kwargs["experiment_id"] == H2ExperimentID.P0_CliffordT:
        answer = []
        pivot_qubits = set()

        # get qubit with highest accumulated measurement error rate
        pivot_qubits.add(noise_model.get_most_noisy_qubit(Op.MEAS)[0])
        if noise_model.basis_gates in [BasisGates.TYPE1, BasisGates.TYPE6]:
            # we get the most noisy qubits in terms of U1 and U2
            pivot_qubits.add(noise_model.get_most_noisy_qubit(Op.U1)[0])
            pivot_qubits.add(noise_model.get_most_noisy_qubit(Op.U2)[0])
        else:
            # we get the most noisy qubits in terms of SX and RZ gates
            pivot_qubits.add(noise_model.get_most_noisy_qubit(Op.SX)[0])
            pivot_qubits.add(noise_model.get_most_noisy_qubit(Op.RZ)[0])
            
        for p in pivot_qubits:
            answer.append({0: p})
        return answer
    else:
        raise Exception("Not implemented!")



if __name__ == "__main__":
    arg_backend = sys.argv[1]
    Precision.PRECISION = MAX_PRECISION
    Precision.update_threshold()
    
    if arg_backend == "gen_configs":
        generate_configs("H2", H2ExperimentID.P0_CliffordT, 4, 7)
    if arg_backend == "embeddings":
        batches = get_num_qubits_to_hardware(WITH_THERMALIZATION)
        
        for num_qubits in batches.keys():
            config_path = get_config_path("H2", H2ExperimentID.P0_CliffordT, num_qubits)
            generate_embeddings(config_path=config_path, experiment_enum=H2ExperimentID, experiment_id=H2ExperimentID.P0_CliffordT, get_hardware_embeddings=get_hardware_embeddings)
    if arg_backend == "test":
        Test.test_hamiltonian()
    
    
    