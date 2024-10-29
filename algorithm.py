from typing import List

from qiskit import QuantumCircuit
from ibm_noise_models import Instruction, instruction_to_ibm


class AlgorithmNode:
    instructions: Instruction

    def __init__(self, action_name: str=None, instruction_sequence=None, next_ins=None, case0=None, case1=None,
                 count_meas=0, serialized=None, actions_to_instructions=None) -> None:
        if serialized is None:
            assert action_name is not None
            assert isinstance(action_name, str)
            assert instruction_sequence is not None
            self.instruction_sequence = instruction_sequence
            self.action_name = action_name
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
            self.depth += 1
            if self.next_ins is None:
                self.is_meas = True
            else:
                self.is_meas = False
        else:
            assert action_name is None
            assert instruction_sequence is None
            assert next_ins is None
            assert case0 is None
            assert case1 is None
            assert actions_to_instructions is not None
            self.action_name = serialized["action"]
            self.instruction_sequence = actions_to_instructions[self.action_name]
            if len(self.instruction_sequence) > 0:
                self.last_target = self.instruction_sequence[-1].target
            
            self.count_meas = 0
            self.depth = 0

            if serialized['next'] == "None":
                self.is_meas = False
                self.next_ins = None
                if serialized['case0'] != "None":
                    self.is_meas = True
                    self.case0 = AlgorithmNode(serialized=serialized['case0'], actions_to_instructions=actions_to_instructions)
                else:
                    self.case0 = None

                if serialized['case1'] != "None":
                    self.is_meas = True
                    self.case1 = AlgorithmNode(serialized=serialized['case1'], actions_to_instructions=actions_to_instructions)
                else:
                    self.case1 = None
                
            else:
                assert serialized['case0'] == "None"
                assert serialized['case1'] == "None"
                self.next_ins = AlgorithmNode(serialized=serialized['next'], actions_to_instructions=actions_to_instructions)
                self.case0 = None
                self.case1 = None
                self.is_meas = False

    def __eq__(self, other: object) -> bool:
        if other is None:
            return False
        return (self.action_name == other.action_name) and (self.next_ins == other.next_ins) and (self.case0 == other.case0) and (self.case1 == other.case1) # WARNING: only comparing action name (not instruction_sequence)
    
    def get_instructions_used(self, current_set) -> List[Instruction]:
        for instruction in self.instruction_sequence:
            if instruction not in current_set:
                current_set.append(instruction)
        
        if self.next_ins is not None:
            assert self.case0 is None
            assert self.case1 is None
            return self.next_ins.get_instructions_used(current_set)
        else:
            if self.case0 is not None:
                self.case0.get_instructions_used(current_set)
            if self.case1 is not None:
                self.case1.get_instructions_used(current_set)
    
    def serialize(self):
        if self.next_ins is not None:
            if self.next_ins.action_name != "halt":
                next_ins = self.next_ins.serialize()
            else:
                next_ins = "None"
        else:
            next_ins = "None"
        
        if self.case0 is not None:
            if self.case0.action_name != "halt":
                case0 = self.case0.serialize()
            else:
                case0 = "None"
        else:
            case0 = "None"
        if self.case1 is not None:
            if self.case1.action_name != "halt":
                case1 = self.case1.serialize()
            else:
                case1 = "None"
        else:
            case1 = "None"

        return {
            "action": self.action_name,
            "next": next_ins,
            "case0": case0,
            "case1": case1,
        }

      
def execute_algorithm(node: AlgorithmNode, qpu: QuantumCircuit, count_ins=0, cbits=None):
    if node is None:
        return count_ins
    
    instruction_to_ibm(qpu, node.instruction_sequence)
    if node.next_ins is not None:
        assert node.case0 is None
        assert node.case1 is None
        return execute_algorithm(node.next_ins, qpu, count_ins+1, cbits=cbits)
    elif node.is_meas:
        with qpu.if_test((cbits[node.last_target], 0)) as else0_:
            execute_algorithm(node.case0, qpu, count_ins+1, cbits=cbits)
        with else0_:
            execute_algorithm(node.case1, qpu, count_ins+1, cbits=cbits)
        return 1
    