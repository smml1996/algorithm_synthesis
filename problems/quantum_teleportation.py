import os, sys
from typing import Dict, List
import qmemory
from qpu_utils import GateData, Op
sys.path.append(os.getcwd()+"/..")

from qstates import QuantumState
from ibm_noise_models import Instruction
import numpy as np
from math import sqrt

class QuantumTeleportationInstance:
    def get_initial_state(self, exp_index: int=None):
        if self.initial_state is None:
            if exp_index == 0:
                self.initial_state = QuantumState(0, dimension=self.num_qubits)
                H0 = GateData(Op.H, self.embedding[0])
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
        rho_initial = initial_state.multi_partial_trace([x for x in range(self.num_qubits) if x!=self.embedding[0]])

        rho_reached = reached_state.multi_partial_trace([x for x in range(self.num_qubits) if x!=self.embedding[0]])

        # compute fidelity:
        arr1 = np.array(rho_initial)
        assert arr1.shape == (2,2)

        arr2 = np.array(rho_reached)
        assert arr2.shape == arr1.shape

        current_fidelity = np.sqrt(np.trace(np.matmul(arr1,arr2)))

        if current_fidelity >= self.fidelity:
            return True
        return False
        


        


        





