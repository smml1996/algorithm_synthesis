from cmath import isclose
from enum import Enum
from math import floor
from typing import Dict, List, Optional
from utils import Precision

P0 = [[1, 0], [0, 0]]
P1 = [[0, 0], [0, 1]]



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
    RX = "RX"
    CZ = "CZ"
    SWAP= "SWAP"

    # MEASUREMENT
    MEAS = "MEASURE"
    P0 = "P0"
    P1 = "P1"

    # NON-UNITARY
    RESET = "RESET"

    # ClassicalOp
    WRITE0 = "WRITE0"
    WRITE1 = "WRITE1"

    DELAY = "DELAY"
    CUSTOM = "CUSTOM"
    def __repr__(self) -> str:
        return self.__str__()
    
class BasisGates(Enum):
    TYPE1 = [Op.CNOT, Op.I, Op.U1, Op.U2, Op.U3]
    TYPE2 = [Op.CNOT, Op.DELAY, Op.I, Op.MEAS, Op.RESET, Op.RZ, Op.SX, Op.X]
    TYPE3 = [Op.CNOT, Op.I, Op.RESET, Op.RZ, Op.SX, Op.X]
    TYPE4 = [Op.CZ, Op.DELAY, Op.I, Op.MEAS, Op.RESET, Op.RZ, Op.SX, Op.X]
    TYPE5 = [Op.I, Op.RZ, Op.SX, Op.X]
    TYPE6 = [Op.CNOT, Op.I, Op.SX, Op.U1, Op.U2, Op.U3, Op.X]
    TYPE7 = [Op.CNOT, Op.I, Op.RZ, Op.SX, Op.X]

def get_basis_gate_type(basis_gates):
    for b in BasisGates:
        if b.value == basis_gates:
            return b
    raise Exception(f"No type matches with the current basis gates ({basis_gates})")
    
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
    if op in [Op.CNOT, Op.CZ, Op.SWAP]:
        return True
    return False
    
def int_to_bin(n: int, zero_padding=None) -> str:
    assert n >= 0
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
    if len(result) == 0:
        return "0"
    return result

def bin_to_int(bin: str) -> int:
    result = 0
    for (power, b) in enumerate(bin):
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
    controls: Optional[int]
    params: Optional[List[float]]

    def __init__(self, label, address, controls=None, params=None) -> None:
        self.label = label
        self.address = address
        self.control = controls
        self.params = params

    def __eq__(self, other: object) -> bool:
        return (self.label, self.address, self.control, self.params) == (
        other.label, other.address, other.control, self.params)

    def __str__(self) -> str:
        d = dict()
        d['gate'] = self.label
        d['address'] = self.address
        d['controls'] = self.control
        d['params'] = self.params
        return d.__str__()

    def __repr__(self) -> str:
        return self.__str__()
    
class NoisyInstruction:
    name: str
    error_free_seq: List[List[GateData]]
    errors: List[List[GateData]]
    probabilities: List[List[float]]

    def __init__(self, name=None, error_free_seq: List[List[GateData]] = None, errors: Dict = None, instructions=None,
                 probabilities: Dict = None, from_serializable=None) -> None:
        if from_serializable is None:
            self.name = name
            self.error_free_seq = error_free_seq
            self.errors = errors
            self.instructions = instructions
            self.probabilities = probabilities
        else:
            self.name = from_serializable['name']
            self.error_free_seq = []
            for g_seq in from_serializable['error_free_seq']:
                seq_obj = []
                for g in g_seq:
                    seq_obj.append(GateData(g['gate'], g['address'], g['controls'], g['params']))
                self.error_free_seq.append(seq_obj)

            self.errors = dict()
            for (error_id, error_seqs) in from_serializable['errors'].items():
                self.errors[error_id] = []

                for seq in error_seqs:
                    seq_obj = []
                    for g in seq:
                        seq_obj.append(GateData(g['gate'], g['address'], g['controls'], g['params']))
                    self.errors[error_id].append(seq_obj)
            self.instructions = from_serializable['instructions']
            self.probabilities = from_serializable['probabilities']
        assert len(self.error_free_seq) == len(self.probabilities.keys())


    def __eq__(self, other: object) -> bool:
        if self.name != other.name:
            return False

        if self.error_free_seq != other.error_free_seq:
            return False

        if self.errors != other.errors:
            return False

        if self.instructions != other.instructions:
            return False

        if self.probabilities != other.probabilities:
            return False

        return True

    def dump_channel(self):
        d = dict()
        d['name'] = self.name
        d['error_free_seq'] = self.error_free_seq
        d['errors'] = self.errors
        d['instructions'] = self.instructions
        d['probabilities'] = self.probabilities
        return d

    def print_channel(self):
        print(f"***** {self.name} *****")
        for i in range(len(self.error_free_seq)):
            print("------")
            print(f"{self.probabilities[i][0]} ~~ {self.error_free_seq[i]}")
            for (j, err) in enumerate(self.errors[i]):
                print(f"{self.probabilities[i][j + 1]} ~~ {err}")

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.__str__()

