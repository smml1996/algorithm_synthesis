from typing import Dict
from qpu_utils import Op, bin_to_int


class ClassicalState:
    sparse_vector: Dict[int, bool]
    def __init__(self, init: Dict[int, bool] = None, is_deepcopy: bool = True):
        if init is None:
            self.sparse_vector = dict()
        else:
            self.sparse_vector = dict()
            for (key, value) in init.sparse_vector.items():
                self.sparse_vector[key] = value

    def __eq__(self, other) -> bool:
        return self.get_memory_val() == other.get_memory_val()

    def __hash__(self):
        return self.get_memory_val()
    
    def get_memory_val(self) -> int:
        answer = 0
        for (key, value) in self.sparse_vector.items():
            answer += (2**key)*value
        return answer
    
    def __str__(self) -> str:
        answer = []
        sorted_keys = sorted(self.sparse_vector.keys())
        for key in sorted_keys:
            answer.append(str(int(self.sparse_vector[key])))
        if len(answer) == 0:
            answer = ['0']
        return str(bin_to_int("".join(answer)))
    
    def __repr__(self):
        return str(self)
    
    def insert(self, key, value):
        self.sparse_vector[key] = value

    

def cwrite(classical_state: ClassicalState, op: Op, address: int) -> ClassicalState:
    result = ClassicalState(classical_state)
    if op == Op.WRITE0:
        result.insert(address, False)
    elif op == Op.WRITE1:
        result.insert(address, True)
    elif op == Op.TOGGLE:
        result.insert(address, not cread(classical_state, address))
    else:
        raise Exception(f"classical write doesnt defined op behaviour for {op}")
    return result


def cread(classical_state: ClassicalState, address: int) -> bool:
    if address in classical_state.sparse_vector.keys():
        return classical_state.sparse_vector[address]
    else:
        return False
    

