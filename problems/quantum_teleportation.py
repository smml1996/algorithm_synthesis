import os, sys
import time
from typing import Dict, List
from cmemory import ClassicalState
from pomdp import build_pomdp
import qmemory
from qpu_utils import GateData, Op, Precision, BasisGates
sys.path.append(os.getcwd()+"/..")

from qstates import QuantumState
from ibm_noise_models import Instruction, NoiseModel, get_ibm_noise_model, HardwareSpec
import numpy as np
from math import pi   
from enum import Enum

# https://pennylane.ai/qml/demos/tutorial_teleportation/

POMDP_OUTPUT_DIR = None
WITH_TERMALIZATION = False
MAX_HORIZON = 10
MIN_HORIZON = 3
MAX_PRECISION = 30
TIME_OUT = 10800 # (in seconds) i.e 3 hours

ALICE_QUBIT = 0
ANCILLA_ALICE = 1
BOB_QUBIT = 2

class ExperimentID(Enum):
    ENTANGLEMENT = 0
    CHANGEBASIS = 1

class QuantumTeleportationInstance:
    """Alice wants to transfer her qubit to BOB

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
    
    def get_initial_state(self, exp_index: int=None):
        if self.initial_state is None:
            if exp_index == 0:
                self.initial_state = QuantumState(0, dimension=self.num_qubits)
                rz_gate = GateData(Op.RX, self.embedding[ALICE_QUBIT], params=[pi/3])
                self.initial_state = qmemory.handle_write(self.initial_state, H0)
            
            raise Exception("experiment index does not exists. unable to create initial state")
        else:
            return self.initial_state

    def __init__(self, num_qubits: int, fidelity: float, instruction_set: List[Instruction], embedding: Dict[int, int]) -> None:
        """ Quantum Teleportation for 1 qubit

        Args:
            num_qubits: number of qubits that the current qpu has, is used to compute density matrices, partial traces.
            fidelity (float): a number between [0,1]

        assume that in 0: we have subsystem A
        assume that in 1: we have subsystem B
        assume that in 2: there is the arbiter
        """
        assert 0 <= fidelity <= 1
        self.num_qubits = num_qubits
        self.fidelity = fidelity
        self.instruction_set = instruction_set
        self.embedding = embedding
        self.initial_state = None
        
    def is_target_state(self, reached_state: QuantumState) -> bool:
        assert self.initial_state is not None
        
        initial_state = self.get_initial_state()
        rho_initial = initial_state.multi_partial_trace([x for x in range(self.num_qubits) if x!=self.embedding[ALICE_QUBIT]])

        rho_reached = reached_state.multi_partial_trace([x for x in range(self.num_qubits) if x!=self.embedding[BOB_QUBIT]])

        # compute fidelity:
        arr1 = np.array(rho_initial)
        assert arr1.shape == (2,2)

        arr2 = np.array(rho_reached)
        assert arr2.shape == arr1.shape

        current_fidelity = np.sqrt(np.trace(np.matmul(arr1,arr2)))

        if current_fidelity >= self.fidelity:
            return True
        return False


def get_embeddings(instruction_set) -> Dict[str, List[Dict[int, int]]]:
    embeddings = dict()
    for hardware_spec in HardwareSpec:
        assert hardware_spec not in embeddings.keys()
        embeddings[hardware_spec] = []
        noise_model = get_ibm_noise_model(hardware_spec)

        if instruction_set == 0:
            pass
    return embeddings
    
def get_experiments_channels(noise_model: NoiseModel, embedding: Dict[int, int], experiment_index: ExperimentID):
    if noise_model.basis_gates == ExperimentID.TYPE5:
        raise Exception("basis gates do not have entanglement gate")
    if experiment_index == ExperimentID.ENTANGLEMENT:
        # alice ancilla needs to get entangled with bob's qubit. Therefore 
        if noise_model.basis_gates in [BasisGates.TYPE1, BasisGates.TYPE6]:
            HAncilla = noise_model.instructions_to_channel[Instruction(embedding[ANCILLA_ALICE], Op.U2, params=[0, pi])]
            SAncilla = noise_model.instructions_to_channel[Instruction(embedding[ANCILLA_ALICE], Op.U1, params=[pi/2])]
            TAncilla = noise_model.instructions_to_channel[Instruction(embedding[ANCILLA_ALICE], Op.U1, params=[pi/4])]

            # Bob corrects his qubits
            if noise_model.basis_gates == BasisGates.TYPE1:
                XBob = noise_model.instructions_to_channel[Instruction(embedding[BOB_QUBIT], Op.U3, params=[pi, 2*pi, pi])]
            else:
                assert noise_model.basis_gates == BasisGates.TYPE6
                XBob = noise_model.instructions_to_channel[Instruction(embedding[BOB_QUBIT], Op.X)]
            ZBob = noise_model.instructions_to_channel[Instruction(embedding[BOB_QUBIT], Op.U1, params=[pi])]
        elif noise_model.basis_gates in [BasisGates.TYPE2, BasisGates.TYPE3, BasisGates.TYPE4, BasisGates.TYPE7]:
            HAncilla = noise_model.instructions_to_channel[Instruction(embedding[ANCILLA_ALICE], Op.SX)]
            SAncilla = noise_model.instructions_to_channel[Instruction(embedding[ANCILLA_ALICE], Op.RZ, params=[pi/2])]
            TAncilla = noise_model.instructions_to_channel[Instruction(embedding[ANCILLA_ALICE], Op.RZ, params=[pi/4])] # https://learning.quantum.ibm.com/tutorial/explore-gates-and-circuits-with-the-quantum-composer#phase-gates

            # Bob corrects his qubits
            XBob = noise_model.instructions_to_channel[Instruction(embedding[BOB_QUBIT], Op.X)]
            ZBob = noise_model.instructions_to_channel[Instruction(embedding[BOB_QUBIT], Op.RZ, params=[pi])]

        if noise_model.basis_gates != BasisGates.TYPE4:
            CX_Anc_Bob = noise_model.instructions_to_channel[Instruction(embedding[BOB_QUBIT], Op.CNOT, control=embedding[ANCILLA_ALICE])]
        else:
            CX_Anc_Bob = noise_model.instructions_to_channel[Instruction(embedding[BOB_QUBIT], Op.CZ, control=embedding[ANCILLA_ALICE])]

        # # now Alice needs to perform a change of basis
        # CX_Alice = noise_model.get_instruction_channel(embedding[ANCILLA_ALICE], Op.CNOT, control=embedding[ALICE_QUBIT])
        # HAlice = noise_model.get_instruction_channel(embedding[ALICE_QUBIT], Op.H)

        # alice qubits are measured 
        meas_alice = noise_model.get_meas_channel(Op.MEAS, embedding[ALICE_QUBIT])
        meas_ancilla = noise_model.get_meas_channel(Op.MEAS, embedding[ANCILLA_ALICE])

        
        return [CX_Anc_Bob]
    else:
        raise Exception(f"No experiments channels specified for {experiment_index} instruction set")
    

def run_experiment_bellman(hardware_spec: str, instruction_set=0):
    assert isinstance(hardware_spec, str)
    # times file
    times_file = open(f"{POMDP_OUTPUT_DIR}times_{hardware_spec}.csv", "w")
    times_file.write("backend,embedding,time\n")


    Precision.PRECISION = MAX_PRECISION
    Precision.update_threshold()

    embeddings = get_embeddings(instruction_set)

    noise_model  = get_ibm_noise_model(hardware_spec.name)
    for (index, embedding) in embeddings[hardware_spec.name]:
        for horizon in range(MIN_HORIZON, MAX_HORIZON):
            print(f"Processing {hardware_spec.name} {index+1}/{len(embeddings[hardware_spec.name])}")
            channels = get_experiments_channels(noise_model, embedding, instruction_set)
            qt_instance = QuantumTeleportationInstance(3, 1.0, channels, embedding)
            initial_state = (qt_instance.get_initial_state(), ClassicalState())
            start_time = time.time()
            pomdp = build_pomdp(channels, initial_state, embedding, horizon)
            end_time = time.time()
            pomdp.serialize(qt_instance.is_target_state, f"{POMDP_OUTPUT_DIR}{hardware_spec.name}_{index}.txt")

            # record time taken to build POMDP
            times_file.write(f"{hardware_spec},{index},{end_time-start_time}\n")
        
        # save embedding
        f = open(f"{POMDP_OUTPUT_DIR}m_{hardware_spec.name}_{index}.txt", "w")
        for i in range(3):
            f.write(f"{i} {embedding[i]}\n")
        f.close()

    times_file.close()
            


    


    
    
    
        


        


        





