import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List
from problems.y_gate import YGateInstance
from experiments_utils import YGateExperimentId, Precision
from ibm_noise_models import Instruction, Op
from numpy import pi
from qstates import QuantumState
import qmemory

def test_instruction_sequence(instance: YGateInstance, instruction_sequence: List[Instruction]):
    current_state = instance.initial_distribution[0][0][0]
    assert isinstance(current_state, QuantumState)
    for instruction in instruction_sequence:
        gate_data = instruction.get_gate_data()
        current_state = qmemory.handle_write(current_state, gate_data)
    
    if current_state != instance.target_state:
        print("target state:", instance.target_state)
        print("current state:", current_state)
        return False
    return True

def test():
    
    embedding = dict({0:0, 1:1})
    instance = YGateInstance(embedding, YGateExperimentId.MAIN99)
    
    instruction_sequences = [
        [Instruction(1, Op.Y)],
        [Instruction(1, Op.U3, params=[pi, pi/2, pi/2])],
        [
            Instruction(1, Op.H), 
            Instruction(1, Op.S), 
            Instruction(1, Op.S),
            Instruction(1, Op.H),
            Instruction(1, Op.S), 
            Instruction(1, Op.S)
        ],
        [
            Instruction(1, Op.S), 
            Instruction(1, Op.S),
            Instruction(1, Op.H), 
            Instruction(1, Op.S), 
            Instruction(1, Op.S),
            Instruction(1, Op.H),
        ],
        [
            Instruction(1, Op.Z),
            Instruction(1, Op.X)
        ],
        [Instruction(1, Op.RY, params=[pi])]
    ]
    
    sequences_names = ["Y", "U3", "HssH", "SshS", "ZX", "RY"]
    for (sequence_name, instruction_sequence) in zip(sequences_names, instruction_sequences):
        print(sequence_name, test_instruction_sequence(instance, instruction_sequence))


if __name__ == "__main__":
    Precision.PRECISION = 8
    Precision.update_threshold()
    test()