import os, sys
sys.path.append(os.getcwd()+"/..")
from qpu_utils import *

if __name__ == "__main__":
    assert is_pauli(Op.X)
    assert is_pauli(Op.Y)
    assert is_pauli(Op.Z)
    assert is_pauli(Op.I)

    assert not is_pauli(Op.MEAS)
    assert not is_pauli(Op.CNOT)
    assert not is_pauli(Op.H)

    get_op("x")
    get_op("I")
    get_op("i")
    get_op("ID")
    get_op("id")
    get_op("y")
    
    assert is_multiqubit_gate(Op.CNOT)
    assert not is_multiqubit_gate(Op.X)
    assert not is_multiqubit_gate(Op.H)
    assert not is_multiqubit_gate(Op.MEAS)