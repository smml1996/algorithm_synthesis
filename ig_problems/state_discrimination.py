from cmath import isclose
from enum import Enum
from typing import Dict
import sys, os


sys.path.append(os.getcwd()+"/..")

from ibm_noise_models import Instruction
import qmemory
from qpu_utils import Op
from qstates import QuantumState
from utils import Precision, are_matrices_equal, find_enum_object
from cmemory import ClassicalState, cread
from experiments_utils import gen_ig_algorithm, rho_qubit0, rho_qubit1
from pomdp import POMDPAction, POMDPVertex


class StateDiscrExperimentID(Enum):
    exp4 = "exp4" # 1 qubit, extra classical space to store outcomes
    exp5 = "exp5" # 1 qubit, discriminate 3 states
    exp6 = "exp6" # 1 qubit, discriminate 4 states
    
    @property
    def exp_name(self):
        return "state_discr"
    
class IGStateDiscrInstance:
    def __init__(self, embedding, experiment_id):
        self.experiment_id = experiment_id
        self.embedding = embedding
        self.initial_states = None
        
        if self.experiment_id in [StateDiscrExperimentID.exp4]:
            assert len(embedding) == 2
            self.hidden_qubit = 1
            self.remove_qubits = [0]
        elif self.experiment_id in [StateDiscrExperimentID.exp5]:
            assert len(embedding) == 3
            self.hidden_qubit = 1
            self.remove_qubits = [0]
        elif self.experiment_id in [StateDiscrExperimentID.exp6]:
            assert len(embedding) == 4
            self.hidden_qubit = 1
        else:
            raise Exception(f"invalid experiment_id {experiment_id}")
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
        
        if self.experiment_id == StateDiscrExperimentID.exp5:
            XNP1 = Instruction(self.hidden_qubit+1, Op.X).get_gate_data()
            X0 = Instruction(0, Op.X).get_gate_data()
            qs1 = qmemory.handle_write(qs0, X0)
            qs1 = qmemory.handle_write(qs1, XNP1)
            self.initial_states.append((qs1, initial_cs))
        
        if self.experiment_id == StateDiscrExperimentID.exp6:
            XNP2 = Instruction(self.hidden_qubit+2, Op.X).get_gate_data()
            X0 = Instruction(0, Op.X).get_gate_data()
            qs_minus = qmemory.handle_write(qs0, X0)
            qs_minus = qmemory.handle_write(qs_minus, H0)
            qs_minus = qmemory.handle_write(qs_minus, XNP2)
            self.initial_states.append((qs_minus, initial_cs))
        
    def is_target_qs(self, hybrid_state):
        qs, cs = hybrid_state
        assert isinstance(qs, QuantumState)
        assert isinstance(cs, ClassicalState)
        if experiment_id == StateDiscrExperimentID.exp4:
            current_rho = qs.multi_partial_trace(remove_indices=self.remove_qubits)
            assert len(current_rho) == 2
            assert len(current_rho[0]) == len(current_rho[1]) 
            assert len(current_rho[0]) == 2
            
            if are_matrices_equal(current_rho, rho_qubit0):
                return cread(cs, self.hidden_qubit) == 1 and cread(cs, self.hidden_qubit+1) == 0
            else:
                assert are_matrices_equal(current_rho, rho_qubit1)
                return cread(cs, self.hidden_qubit) == 0 and cread(cs, self.hidden_qubit+1) == 1
        elif experiment_id == StateDiscrExperimentID.exp5:
            current_rho0 = qs.multi_partial_trace(remove_indices=[0,2])
            current_rho1 = qs.multi_partial_trace(remove_indices=[0,1])
            if are_matrices_equal(current_rho0, rho_qubit0):
                if are_matrices_equal(current_rho1, rho_qubit1):
                    return cread(cs, self.hidden_qubit) == 1 and cread(cs, self.hidden_qubit+1) == 1
                else:
                    return cread(cs, self.hidden_qubit) == 1 and cread(cs, self.hidden_qubit+1) == 0
            else:
                assert are_matrices_equal(current_rho0, rho_qubit1)
                assert are_matrices_equal(current_rho1, rho_qubit0)
                return cread(cs, self.hidden_qubit) == 0 and cread(cs, self.hidden_qubit+1) == 1
            
        elif experiment_id == StateDiscrExperimentID.exp6:
            current_rho0 = qs.multi_partial_trace(remove_indices=[0,2,3])
            current_rho1 = qs.multi_partial_trace(remove_indices=[0,1,3])
            current_rho2 = qs.multi_partial_trace(remove_indices=[0,1,2])
            if are_matrices_equal(current_rho0, rho_qubit0):
                if are_matrices_equal(current_rho1, rho_qubit1):
                    are_matrices_equal(current_rho2, rho_qubit0)
                    return cread(cs, self.hidden_qubit) == 1 and cread(cs, self.hidden_qubit+1) == 1 and cread(cs, self.hidden_qubit+2) == 0
                elif are_matrices_equal(current_rho2, rho_qubit1):
                    return cread(cs, self.hidden_qubit) == 0 and cread(cs, self.hidden_qubit+1) == 0 and cread(cs, self.hidden_qubit+2) == 1
                else:
                    return cread(cs, self.hidden_qubit) == 1 and cread(cs, self.hidden_qubit+1) == 0 and cread(cs, self.hidden_qubit+2) == 0
            else:
                assert are_matrices_equal(current_rho0, rho_qubit1)
                assert are_matrices_equal(current_rho1, rho_qubit0)
                assert are_matrices_equal(current_rho2, rho_qubit0)
                return cread(cs, self.hidden_qubit) == 0 and cread(cs, self.hidden_qubit+1) == 1 and cread(cs, self.hidden_qubit+2) == 0
        else:
            raise Exception(f"invalid experiment id {experiment_id}")
    
def get_experiments_actions(embedding, experiment_id):
    assert isinstance(experiment_id, StateDiscrExperimentID)
    
    if experiment_id in [StateDiscrExperimentID.exp4]:
    
        num_qubits = 1
        hidden_qubit = 1
        assert len(embedding.keys()) == 2
        
        actions = []
        
        # all clifford gates
        for i in range(num_qubits):
            assert experiment_id in [StateDiscrExperimentID.exp5, StateDiscrExperimentID.exp4]
            ops = [Op.H, Op.MEAS]
            
            for op in ops:
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
    elif experiment_id in [StateDiscrExperimentID.exp5, StateDiscrExperimentID.exp6]:
        num_qubits = 1
        hidden_qubit = 1
        if experiment_id == StateDiscrExperimentID.exp5:
            assert len(embedding.keys()) == 3
        else:
            assert len(embedding.keys()) == 4
            
        
        actions = []
        
        # all clifford gates
        for i in range(num_qubits):
            assert experiment_id in [StateDiscrExperimentID.exp5, StateDiscrExperimentID.exp6]
            ops = [Op.H, Op.MEAS]
            
            for op in ops:
                actions.append(POMDPAction(f"{op.name}{i}", [Instruction(i, op)]))
            for j in range(i+1, num_qubits):
                actions.append(POMDPAction(f"CX{i}{j}", [Instruction(j, Op.CNOT, i)]))
        
        if experiment_id == StateDiscrExperimentID.exp5:
            RESET = POMDPAction("RESET", [Instruction(embedding[0], Op.RESET),
                                        Instruction(embedding[0], Op.CH, hidden_qubit),
                                        Instruction(embedding[0], Op.CNOT, hidden_qubit+1),
                                        ])
        else:
            RESET = POMDPAction("RESET", [Instruction(embedding[0], Op.RESET),
                                        Instruction(embedding[0], Op.CH, hidden_qubit),
                                        Instruction(embedding[0], Op.CNOT, hidden_qubit+1),
                                        Instruction(embedding[0], Op.CNOT, hidden_qubit+2),
                                        Instruction(embedding[0], Op.CH, hidden_qubit+2),
                                        ])
        DETERMINE0 = POMDPAction("IS0", [Instruction(hidden_qubit, Op.WRITE1)])
        DETERMINEPlus = POMDPAction("ISPlus", [Instruction(hidden_qubit+1, Op.WRITE1)])
        DETERMINE1 = POMDPAction("IS1", [Instruction(hidden_qubit, Op.WRITE1), Instruction(hidden_qubit+1, Op.WRITE1)])
        actions.append(RESET)
        actions.append(DETERMINE0)
        actions.append(DETERMINEPlus)
        actions.append(DETERMINE1)
        
        if experiment_id == StateDiscrExperimentID.exp6:
            DETERMINEMinus = POMDPAction("ISMinus", [Instruction(hidden_qubit+2, Op.WRITE1)])
            actions.append(DETERMINEMinus)
        print("******")
        for action in actions:
            print(action.name)
        return actions

    else:
        raise Exception(f"experiment actions not defined for experiment {experiment_id}")
    
def get_experiment_id_qubits(experiment_id) -> int:
    if experiment_id in [StateDiscrExperimentID.exp5]:
        return 3
    elif experiment_id == StateDiscrExperimentID.exp6:
        return 4
    elif experiment_id == StateDiscrExperimentID.exp4:
        return 2
    else:
        raise Exception(f"Invalid experiment id {experiment_id}")

def get_experiment_id_horizon(experiment_id) -> int:
    return -1

def guard(vertex: POMDPVertex, embedding: Dict[int, int], action: POMDPAction) -> bool:
    cs = vertex.classical_state
    
    hidden_qubit = len(embedding.keys()) - 1
    
    if cread(cs, hidden_qubit) == 1 and cread(cs, hidden_qubit+1) == 1:
        return action.name == "RESET"
    if cread(cs, hidden_qubit) == 0 and cread(cs, hidden_qubit+1) == 1:
        return False
    if cread(cs, hidden_qubit) == 1 and cread(cs, hidden_qubit+1) == 1:
        return False
    if cread(cs, hidden_qubit) == 1 and cread(cs, hidden_qubit+1) == 0:
        return False
    if cread(cs, hidden_qubit + 2) != 0:
        
        return False
    return True

def get_guard(experiment_id):
    return guard

if __name__ == "__main__":
    arg_exp = sys.argv[1]
    experiment_id = find_enum_object(arg_exp, StateDiscrExperimentID)
    if experiment_id is None:
        raise Exception(f"invalid argument: {arg_exp}")
    gen_ig_algorithm(experiment_id, 
                    get_experiment_id_qubits(experiment_id),
                    get_experiments_actions,
                    IGStateDiscrInstance,
                    get_guard(experiment_id),
                    get_experiment_id_horizon(experiment_id))

    
