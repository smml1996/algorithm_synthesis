from typing import List, Set

from qiskit import QuantumCircuit
from ibm_noise_models import Instruction, instruction_to_ibm
from pomdp import POMDPAction
from random import random


class AlgorithmNode:
    instructions: Instruction

    def __init__(self, action_name: str=None, instruction_sequence=None, classical_state=None,children=None, serialized=None, actions_to_instructions=None, noiseless=False, children_probs=dict()) -> None:
        self.noiseless = noiseless
        if serialized is None:
            assert action_name is not None
            assert isinstance(action_name, str)
            assert instruction_sequence is not None
            assert classical_state is not None
            self.instruction_sequence = instruction_sequence
            self.action_name = action_name
            self.children = children
            self.classical_state = classical_state
            self.children_probs = children_probs
        else:
            assert action_name is None
            assert instruction_sequence is None
            assert actions_to_instructions is not None
            self.action_name = serialized["action"]
            self.instruction_sequence = actions_to_instructions[self.action_name]
            self.classical_state = serialized["classical_state"]
            self.children = []
            self.children_probs = serialized["children_probs"]
            for child in serialized["children"]:
                self.children.append(AlgorithmNode(serialized=child, actions_to_instructions=actions_to_instructions))

    def __eq__(self, other: object) -> bool:
        if other is None:
            return self.action_name == "halt"
        
        # WARNING: only comparing action name (not instruction_sequence)
        if self.action_name == other.action_name and self.classical_state == other.classical_state and len(self.children) == len(other.children):
            for child in self.children:
                if not (child in other.children):
                    return False
            return True
        return False
    
    def serialize(self, depth=0):
        if self.children is not None:
            children = []
            for child in self.children:
                children.append(child.serialize(depth=depth+1))
        else:
            children = "None"

        return {
            "action": self.action_name,
            "classical_state": self.classical_state,
            "children": children,
            "depth": depth
        }
    
    def get_algorithm_actions(self) -> Set[str]:
        if self.action_name != "halt":
            result = set([self.action_name])
        else:
            result = set()
        
        for child in self.children:
            temp = child.get_algorithm_actions()
            for action in temp:
                result.add(action)
        return result
      
def execute_algorithm(node: AlgorithmNode, qpu: QuantumCircuit, count_ins=0, cbits=None):    
    if node is not None:
        if len(node.children_probs.keys()) > 0:
            elements = []
            probs = []
            
            assert(len(node.children) == len(node.children_probs))
            
            for (element, prob) in node.children_probs.items():
                elements.append(element)
                probs.append(prob)
                
            next_node_index = random.choices(elements, weights=probs, k=1)[0]
            execute_algorithm(node.children[next_node_index], qpu, count_ins, cbits=cbits)
        else:
            instruction_to_ibm(qpu, node.instruction_sequence, noiseless=node.noiseless)
            for child in node.children:
                if len(node.children) > 1:
                    with qpu.if_test((cbits, child.classical_state)):
                        execute_algorithm(child, qpu, count_ins+1, cbits=cbits)
                else:
                    execute_algorithm(child, qpu, count_ins+1, cbits=cbits)
    
def get_algorithm(current_node, tabs="\t"):
    if current_node is None:
        return f"{tabs}pass\n"
    assert isinstance(current_node, AlgorithmNode)
    result = f"{tabs}instruction_to_ibm(qc, basis_gates, {current_node.action_name})\n"
    for child in current_node.children:
        if len(current_node.children) > 1:
            result += f"{tabs}with qc.if_test((cbits, {child.classical_state})):\n"
            child_alg = get_algorithm(child, tabs= f"{tabs}\t")
        else:
            child_alg = get_algorithm(child, tabs= f"{tabs}")
        result += child_alg
    return result

def dump_algorithms(algorithms: List[AlgorithmNode], actions: List[POMDPAction], output_path, comments=None):
    assert (comments is None) or (len(comments) == len(algorithms))
    file = open(output_path, "w")
    file.write("import os, sys\n")
    file.write("sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n")
    file.write("from qiskit import QuantumCircuit, ClassicalRegister\n")
    file.write("from ibm_noise_models import Instruction, instruction_to_ibm, Op\n\n")
    
    file.write("###### ACTIONS ######\n")
    for action in actions:
        assert isinstance(action, POMDPAction)
        instructions_str = ""
        for instruction in action.instruction_sequence:
            if len(instructions_str) > 0:
                instructions_str += ", "
            instructions_str += f"Instruction({instruction.target}, Op.{instruction.op.name}, {instruction.control}, {instruction.params})"
        file.write(f"{action.name} = [{instructions_str}]\n")
    file.write("halt = []\n")
    
    file.write("###### END ACTIONS ######\n\n")
    
    for (index, algorithm) in enumerate(algorithms):
        assert isinstance(algorithm, AlgorithmNode)
        file.write(f"def algorithm{index}(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):\n")
        if comments is not None:
            initial_comment = comments[index]
            file.write(f"\t\'\'\'{initial_comment}\'\'\'\n")
        file.write(get_algorithm(algorithm))
        file.write("\n\n")

    file.write("algorithms = []\n")
    for i in range(len(algorithms)):
        file.write(f"algorithms.append(algorithm{i})\n")
    file.close()
