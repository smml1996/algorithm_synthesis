from importlib.machinery import SourceFileLoader
from typing import List

from qiskit import QuantumCircuit
from ibm_noise_models import Instruction, instruction_to_ibm


class AlgorithmNode:
    instructions: Instruction

    def __init__(self, instruction=None, next_ins=None, case0=None, case1=None,
                 count_meas=0, serialized=None) -> None:
        if serialized is None:
            assert instruction is not None
            assert isinstance(instruction, Instruction)
            self.next_ins = next_ins
            self.case0 = case0
            self.case1 = case1
            self.count_meas = count_meas
            self.depth = 0
            if self.next_ins is not None:
                self.depth = self.next_ins
            if case1 is not None:
                assert case0 is not None
                self.depth = max(case1.depth, case0.depth)
            if instruction != Instruction.I:
                self.depth += 1
        else:
            assert instruction is None
            assert next_ins is None
            assert case0 is None
            assert case1 is None
            self.instruction = Instruction(serialized['instruction']['target'], serialized['instruction']['op'], serialized['instruction']['control'], serialized[instruction]['params'])
            # self.count_meas = serialized['count_meas']
            self.count_meas = 0
            # self.depth = serialized['depth']
            self.depth = 0

            if serialized['next'] is None:
                self.next_ins = None
                if serialized['case0'] is not None:
                    self.case0 = AlgorithmNode(serialized=serialized['case0'])
                else:
                    self.case0 = None

                if serialized['case1'] is not None:
                    self.case1 = AlgorithmNode(serialized=serialized['case1'])
                else:
                    self.case1 = None
            else:
                assert serialized['case0'] is None
                assert serialized['case1'] is None
                self.next_ins = AlgorithmNode(serialized=serialized['next'])
                self.case0 = None
                self.case1 = None

    def __eq__(self, other: object) -> bool:
        if other is None:
            return False
        return (self.instruction == other.instruction) and (self.next_ins == other.next_ins) and (self.case0 == other.case0) and (self.case1 == other.case1)
    
    def get_instructions_used(self, current_set) -> List[Instruction]:
        current = []
        if self.instruction != Instruction.op.I:
            current = self.instruction
            if current not in current_set:
                current_set.append(current)
        
        if self.next_ins is not None:
            assert self.case0 is None
            assert self.case1 is None
            return self.next_ins.get_instructions_used(current_set)
        else:
            if self.case0 is not None:
                self.case0.get_instructions_used(current_set)
            if self.case1 is not None:
                self.case1.get_instructions_used(current_set)
    
    def serialize(self, for_json=False):
        if for_json:
            next_ins = -1
        else:
            next_ins = None
        if self.next_ins is not None:
            next_ins = self.next_ins.serialize(for_json)
        
        if for_json:
            case0 = -1
            case1 = -1
        else:
            case0 = None
            case1 = None
        if self.case0 is not None:
            case0 = self.case0.serialize(for_json)
        if self.case1 is not None:
            case1 = self.case1.serialize(for_json)

        instruction = self.instruction
        if for_json:
            instruction = instruction.name
        return {
            "instruction": instruction.serialize(for_json=for_json),
            "next": next_ins,
            "case0": case0,
            "case1": case1,
            "count_meas": self.count_meas,
            "depth": self.depth
        }

      
def execute_algorithm(node: AlgorithmNode, qpu, count_ins=0, basis_gates=None, cbits=None, address_space=None):
    if node is None:
        return count_ins
    
    assert basis_gates is not None
    assert cbits is not None
    instruction_to_ibm(qpu, basis_gates, node.instruction)
    assert address_space is None
    if node.next_ins is not None:
        assert node.case0 is None
        assert node.case1 is None
        return execute_algorithm(node.next_ins, qpu, count_ins+1, basis_gates=basis_gates, cbits=cbits)
    elif node.instruction.is_meas_instruction():
        with qpu.if_test((cbits[node.instruction.target], 0)) as else0_:
            execute_algorithm(node.case0, qpu, count_ins+1, basis_gates=basis_gates, cbits=cbits)
        with else0_:
            execute_algorithm(node.case1, qpu, count_ins+1, basis_gates=basis_gates,cbits=cbits)
        return 1
    
def load_algorithms_file(path) -> List[AlgorithmNode]:
    mod = SourceFileLoader("m", path).load_module()
    result = []
    for algorithm in mod.algorithms:
        if algorithm is not None:
            result.append(AlgorithmNode(serialized=algorithm))
    return result