from enum import Enum


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

    # MEASUREMENT
    MEAS = "MEASURE"

    # NON-UNITARY
    RESET = "RESET"

    # ClassicalOp
    WRITE0 = "WRITE0"
    WRITE1 = "WRITE1"

    DELAY = "DELAY"

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
    if op in [Op.CNOT, Op.RZ, Op.CZ]:
        return True
    return False
    
