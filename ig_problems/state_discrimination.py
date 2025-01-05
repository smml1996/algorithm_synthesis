from enum import Enum
from typing import Dict
import sys, os


sys.path.append(os.getcwd()+"/..")

from ibm_noise_models import Instruction
import qmemory
from qpu_utils import Op
from qstates import QuantumState
from utils import are_matrices_equal, find_enum_object
from cmemory import ClassicalState, cread
from experiments_utils import gen_ig_algorithm, rho_qubit0, rho_qubit1
from pomdp import POMDPAction, POMDPVertex


class StateDiscrExperimentID(Enum):
    exp1 = "exp1" # 2 qubits, all possible clifford gates between these two qubits
    exp2 = "exp2"
    
    @property
    def exp_name(self):
        return "state_discr"
    
class IGStateDiscrInstance:
    def __init__(self, embedding, experiment_id):
        self.experiment_id = experiment_id
        self.embedding = embedding
        self.initial_states = None
        if self.experiment_id == StateDiscrExperimentID.exp1:
            assert len(embedding) == 3
            self.hidden_qubit = 2
            self.remove_qubits = [0,1]
        else:
            assert self.experiment_id == StateDiscrExperimentID.exp2
            self.remove_qubits = [0,1,2]
            self.hidden_qubit = 3
        self.get_initial_states()
        
    def get_initial_states(self):
        self.initial_states = []
        initial_cs = ClassicalState()
        H0 = Instruction(self.embedding[0], Op.H).get_gate_data()
        XN = Instruction(self.hidden_qubit, Op.X).get_gate_data()
        qs0 = QuantumState(0, qubits_used=list(self.embedding.values()))
        self.initial_states.append((qs0, initial_cs))
        qs_plus = qmemory.handle_write(qs0, H0)
        qs_plus = qmemory.handle_write(qs_plus, XN)
        self.initial_states.append((qs_plus, initial_cs))
        
    def is_target_qs(self, hybrid_state):
        qs, cs = hybrid_state
        assert isinstance(qs, QuantumState)
        assert isinstance(cs, ClassicalState)
        current_rho = qs.multi_partial_trace(remove_indices=self.remove_qubits)
        assert len(current_rho) == 2
        assert len(current_rho[0]) == len(current_rho[1]) 
        assert len(current_rho[0]) == 2
        
        if are_matrices_equal(current_rho, rho_qubit0):
            return cread(cs, self.hidden_qubit) == 1 and cread(cs, self.hidden_qubit+1) == 0
        else:
            assert are_matrices_equal(current_rho, rho_qubit1)
            return cread(cs, self.hidden_qubit) == 0 and cread(cs, self.hidden_qubit+1) == 1
    
def get_experiments_actions(embedding, experiment_id):
    assert isinstance(experiment_id, StateDiscrExperimentID)
    
    if experiment_id in [StateDiscrExperimentID.exp1, StateDiscrExperimentID.exp2]:
        if experiment_id == StateDiscrExperimentID.exp1:
            num_qubits = 2
            hidden_qubit = 2
            assert len(embedding.keys()) == 3
        else:
            num_qubits = 3
            hidden_qubit = 3
            assert len(embedding.keys()) == 4
        
        actions = []
        
        # all clifford gates
        for i in range(num_qubits):
            for op in [Op.H, Op.S, Op.MEAS]:
                actions.append(POMDPAction(f"{op.name}{i}", [Instruction(i, op)]))
            for j in range(i+1, num_qubits):
                actions.append(POMDPAction(f"CX{i}{j}", [Instruction(j, Op.CNOT, i)]))
        
        RESET = POMDPAction("RESET", [Instruction(embedding[0], Op.RESET),
                                      Instruction(embedding[0], Op.CH, hidden_qubit),
                                      ])
        DETERMINE0 = POMDPAction("IS0", [Instruction(hidden_qubit, Op.WRITE1)])
        DETERMINEPlus = POMDPAction("ISPlus", [Instruction(hidden_qubit+1, Op.WRITE1)])
        actions.append(RESET)
        actions.append(DETERMINE0)
        actions.append(DETERMINEPlus)
        print("******")
        for action in actions:
            print(action.name)
        return actions
    else:
        raise Exception(f"experiment actions not defined for experiment {experiment_id}")
    
def get_experiment_id_qubits(experiment_id) -> int:
    if experiment_id == StateDiscrExperimentID.exp1:
        return 3
    else:
        assert experiment_id == StateDiscrExperimentID.exp2
        return 4

def get_experiment_id_horizon(experiment_id) -> int:
    return -1

def guard(vertex: POMDPVertex, embedding: Dict[int, int], action: POMDPAction) -> bool:
    cs = vertex.classical_state
    
    hidden_qubit = 2
    
    if cread(cs, hidden_qubit) == 1 and cread(cs, hidden_qubit+1) == 1:
        return action.name == "RESET"
    if cread(cs, hidden_qubit) == 0 and cread(cs, hidden_qubit+1) == 1:
        return False
    if cread(cs, hidden_qubit) == 1 and cread(cs, hidden_qubit+1) == 0:
        return False
    
    return True

if __name__ == "__main__":
    arg_exp = sys.argv[1]
    experiment_id = find_enum_object(arg_exp, StateDiscrExperimentID)
    if experiment_id is None:
        raise Exception(f"invalid argument: {arg_exp}")
    gen_ig_algorithm(experiment_id, 
                    get_experiment_id_qubits(experiment_id),
                    get_experiments_actions,
                    IGStateDiscrInstance,
                    guard,
                    get_experiment_id_horizon(experiment_id))

    
