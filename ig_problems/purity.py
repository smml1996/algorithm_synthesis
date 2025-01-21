from cmath import isclose
from enum import Enum
import sys, os
from typing import Dict


sys.path.append(os.getcwd()+"/..")

from ibm_noise_models import Instruction
import qmemory
from qpu_utils import Op
from qstates import QuantumState
from utils import Precision, are_matrices_equal, find_enum_object
from cmemory import ClassicalState, cread
from experiments_utils import gen_ig_algorithm, rho_qubit0, rho_qubit1, default_guard
from pomdp import POMDPAction, POMDPVertex

class PurityExperimentID(Enum):
    isbell = "isbell" # 2 qubits (0.5, 1.00)
    
    @property
    def exp_name(self):
        return "purity"
    
class IGPurityInstance:
    def __init__(self, embedding, experiment_id):
        self.experiment_id = experiment_id
        self.embedding = embedding
        self.initial_states = None
        
        if self.experiment_id in [PurityExperimentID.isbell]:
            assert len(embedding) == 3 # hidden qubit is third qubit
            self.hidden_qubit = 2
        else:
            raise Exception(f"invalid experiment_id {experiment_id}")
        self.get_initial_states()
        
    def get_initial_states(self):
        self.initial_states = []
        initial_cs = ClassicalState()
        H0 = Instruction(self.embedding[0], Op.H).get_gate_data()
        CX = Instruction(self.embedding[1], Op.CNOT, self.embedding[0]).get_gate_data()
        XN = Instruction(self.hidden_qubit, Op.X).get_gate_data()
        qs0 = QuantumState(0, qubits_used=list(self.embedding.values()))
        self.initial_states.append((qs0, initial_cs)) # purity 1 |00>
        
        qs_1 = qmemory.handle_write(qs0, H0)
        qs_1 = qmemory.handle_write(qs_1, CX)
        qs_1 = qmemory.handle_write(qs_1, XN)
        self.initial_states.append((qs_1, initial_cs)) # Bell state |01>
        
        
    def is_target_qs(self, hybrid_state):
        qs, cs = hybrid_state
        assert isinstance(qs, QuantumState)
        assert isinstance(cs, ClassicalState)
        
        rho = qs.multi_partial_trace(remove_indices=[0,1])
        
        if are_matrices_equal(rho, rho_qubit0):
            # check classical state is 0,1
            return cread(cs, 3)
        else:
            assert are_matrices_equal(rho, rho_qubit1)
            # check classical state is 1,0
            return cread(cs, 2) == 1
    
def get_experiments_actions(embedding, experiment_id):
    assert isinstance(experiment_id, PurityExperimentID)

    num_qubits = len(embedding.keys()) - 1
    hidden_qubit = num_qubits
    actions = []
    print("actions", hidden_qubit)
    if experiment_id == PurityExperimentID.isbell:
        assert len(embedding) == 3
        
        # all clifford gates
        for i in range(num_qubits):
            ops = [Op.MEAS, Op.H]
            
            for op in ops:
                actions.append(POMDPAction(f"{op.name}{i}", [Instruction(i, op)]))
            for j in range(num_qubits):
                if i != j:
                    actions.append(POMDPAction(f"CX{i}{j}", [Instruction(j, Op.CNOT, i)]))
        
        RESET = POMDPAction("RESET", [Instruction(embedding[0], Op.RESET),
                                      Instruction(embedding[1], Op.RESET),
                                      Instruction(embedding[0], Op.CH, hidden_qubit),
                                      Instruction(embedding[1], Op.CNOT, embedding[0])
                                      ])
        NOTEntangled = POMDPAction("NOTEntangled", [Instruction(hidden_qubit, Op.WRITE0),Instruction(hidden_qubit+1, Op.WRITE1)])
        Entangled = POMDPAction("Entangled", [Instruction(hidden_qubit, Op.WRITE1), Instruction(hidden_qubit+1, Op.WRITE0)])
        actions.append(RESET)
        actions.append(NOTEntangled)
        actions.append(Entangled)
    else:
        raise Exception(f"invalid experiment {experiment_id}")
    
    for action in actions:
        print(action.name)
    return actions
        
def guard(vertex: POMDPVertex, embedding: Dict[int, int], action: POMDPAction) -> bool:
    cs = vertex.classical_state
    return cread(cs, 2) == 0 and cread(cs, 3) == 0       
          
def get_experiment_id_qubits(experiment_id) -> int:
    if experiment_id in [PurityExperimentID.isbell]:
        return 3
    else:
        raise Exception(f"Invalid experiment id {experiment_id}")
    
def get_experiment_id_horizon(experiment_id) -> int:
    return -1

def get_guard(experiment_id):
    return guard

if __name__ == "__main__":
    arg_exp = sys.argv[1]
    experiment_id = find_enum_object(arg_exp, PurityExperimentID)
    if experiment_id is None:
        raise Exception(f"invalid argument: {arg_exp}")
    gen_ig_algorithm(experiment_id, 
                    get_experiment_id_qubits(experiment_id),
                    get_experiments_actions,
                    IGPurityInstance,
                    get_guard(experiment_id),
                    get_experiment_id_horizon(experiment_id))