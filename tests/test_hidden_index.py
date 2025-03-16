import os, sys

sys.path.append(os.getcwd()+"/..")
import qmemory
from utils import Precision
from qstates import QuantumState
from cmemory import ClassicalState
from qpu_utils import Op
from ibm_noise_models import HardwareSpec, Instruction, NoiseModel
from pomdp import POMDP, POMDPAction, build_pomdp, default_guard

def test_hidden_index(with_hidden_index):
    print(f"Test: with_hidden_index={with_hidden_index}")
    noise_model = NoiseModel(hardware_specification=HardwareSpec.ATHENS, thermal_relaxation=False)
    embedding = {0:0}
    
    X0 = Instruction(0, Op.X).get_gate_data()
    classical_state = ClassicalState()
    quantum_state = QuantumState(0, qubits_used=[0])
    initial_distribution = [((quantum_state, classical_state), 0.5)]
    quantum_state = qmemory.handle_write(quantum_state, X0)
    initial_distribution.append(((quantum_state, classical_state), 0.5))
    
    H0_instruction = Instruction(0, Op.H).to_basis_gate_impl(noise_model.basis_gates)
    S0_instruction = Instruction(0, Op.S).to_basis_gate_impl(noise_model.basis_gates)
    meas_instruction = Instruction(0, Op.MEAS).to_basis_gate_impl(noise_model.basis_gates)
    
    actions = [
        POMDPAction("H0", H0_instruction),
        POMDPAction("S0", S0_instruction),
        POMDPAction("MEAS", meas_instruction)
    ]
    
    
    pomdp = build_pomdp(actions, noise_model, 4, embedding, initial_distribution=initial_distribution, guard=default_guard, qubits_used=set([0]), set_hidden_index=with_hidden_index)
    
    assert isinstance(pomdp, POMDP)
    
    assert pomdp.initial_state.hidden_index is None
    
    hidden_indices = set()
    for (source_v, dict_source_v) in pomdp.transition_matrix.items():
        if source_v != pomdp.initial_state:
            hidden_indices.add(source_v.hidden_index)
            for (channel, dict_channel) in dict_source_v.items():
                for target in dict_channel.keys():
                    if with_hidden_index:
                        assert source_v.hidden_index is not None
                        assert source_v.hidden_index == 0 or source_v.hidden_index == 1
                        
                    else:
                        assert source_v.hidden_index is None
                    if source_v.hidden_index != target.hidden_index:
                        raise Exception(f"hidden indices do not match ({source_v.hidden_index}!={target.hidden_index})")
    if with_hidden_index:
        assert len(hidden_indices) == 2
    else:
        assert len(hidden_indices) == 1

if __name__ == "__main__":
    Precision.PRECISION = 8
    Precision.update_threshold()
    test_hidden_index(with_hidden_index=True)
    test_hidden_index(with_hidden_index=False)

