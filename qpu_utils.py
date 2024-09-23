from cmath import isclose
from enum import Enum
from math import floor
from typing import List, Optional

class Precision:
    PRECISION = 8  # round number to `PRECISION` floating point digits
    isclose_abstol = None
    rel_tol = None
    is_lowerbound = True
    @staticmethod
    def update_threshold():
        Precision.isclose_abstol = 0.00000000001
        Precision.rel_tol = 1/(10**(Precision.PRECISION-1))  

class Op(Enum):
    # PAULI GATES
    X = "X"
    Y = "Y"
    Z = "Z"
    I = "I"

    SX = "SX"
    U3 = "U3"
    U2 = "U2"
    U1 = "U1"

    # HADAMARD
    H = "H"

    # MULTI-QUBIT GATES
    CNOT = "CNOT"
    RZ = "RZ"
    CZ = "CZ"
    SWAP= "SWAP"

    # MEASUREMENT
    MEAS = "MEASURE"

    # NON-UNITARY
    RESET = "RESET"

    # ClassicalOp
    WRITE0 = "WRITE0"
    WRITE1 = "WRITE1"

    DELAY = "DELAY"
    CUSTOM = "CUSTOM"
    def __repr__(self) -> str:
        return self.__str__()
    
def is_pauli(op: Op):
    return op in [Op.X, Op.Z, Op.Y, Op.I]
    
def get_op(op_: str) -> Op:
    '''used to get an Operator (Enum defined above) with name op_
    '''
    op_ = op_.strip().upper()
    if op_ == "CX":
        op_ = "CNOT"
    if op_ == "ID":
        op_ = "I"
    for op in Op:
        if op.value == op_:
            return op
    raise Exception("Could not retrieve operator", op_)

def is_multiqubit_gate(op: Op):
    assert isinstance(op, Op)
    if op in [Op.CNOT, Op.RZ, Op.C, Op.SWAP]:
        return True
    return False
    
def int_to_bin(n: int, zero_padding=None) -> str:
    assert n >= 0
    if n == 0:
        return "0"
    result = ""
    while n > 0:
        if (n % 2) == 1:
            result += "1"
        else:
            result += "0"
        n = floor(n / 2)

    if zero_padding is not None:
        while len(result) < zero_padding:
            result += "0"
    return result

def bin_to_int(bin: str) -> int:
    result = 0
    for (b, power) in enumerate(bin):
        assert b=="0" or b == "1"
        result += int(b)*(2**power)
    return result 


def get_complex(amplitude):
    if not isinstance(amplitude, complex):
        amplitude = complex(amplitude, 0.0)
    if isinstance(amplitude, complex):
        if isclose(amplitude.real, 0.0, abs_tol=Precision.isclose_abstol):
            amplitude = complex(0.0, amplitude.imag)
        if isclose(amplitude.imag, 0.0, abs_tol=Precision.isclose_abstol):
            amplitude = complex(amplitude.real, 0.0)
    return amplitude

def get_num_controls(op: Op) -> int:
    assert is_multiqubit_gate(op)
    if op == Op.CNOT:
        return 1
    elif op == Op.CU3:
        return 1
    elif op == Op.CZ:
        return 1
    else:
        raise Exception(f"multiqubit op not implemented ({op})")
    
def is_projector(op: Op) -> bool:
    if op in [Op.MEAS]:
        return True
    return False
    
class GateData: # this was previously called GateData
    label: Op
    address: int
    controls: Optional[List[int]]
    params: Optional[List[float]]

    def __init__(self, label, address, controls=None, params=None) -> None:
        self.label = label
        self.address = address
        self.controls = controls
        self.params = params

    def __eq__(self, other: object) -> bool:
        return (self.label, self.address, self.controls, self.params) == (
        other.label, other.address, other.controls, self.params)

    def __str__(self) -> str:
        d = dict()
        d['gate'] = self.label
        d['address'] = self.address
        d['controls'] = self.controls
        d['params'] = self.params
        return d.__str__()

    def __repr__(self) -> str:
        return self.__str__()
