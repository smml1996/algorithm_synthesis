import os, sys
sys.path.append(os.getcwd()+"/..")
from ibm_noise_models import *

def test_instruction_equality():
    d = dict()
    instruction1 = Instruction(1, Op.X, None)
    d[instruction1] = 1
    assert len(d.keys()) == 1 
    instruction2 = Instruction(1, Op.X, None)
    d[instruction2] = 0
    assert len(d.keys()) == 1

    instruction3 = Instruction(1, Op.Z, None)
    d[instruction3] = 9
    assert len(d.keys()) == 2

    assert instruction1 == instruction1
    assert instruction1 == instruction2

    try:
        instruction3 = Instruction(1, Op.X, 1)
        assert False
    except:
        pass

    try:
        instruction3 = Instruction(1, Op.CNOT, 1)
        assert False
    except:
        pass

    instruction3 = Instruction(1, Op.CNOT, 2)
    d[instruction3] = 10
    assert len(d.keys()) == 3
    instruction4 = Instruction(1, Op.CNOT, 2)
    d[instruction4] = 11
    assert len(d.keys()) == 3


    assert instruction1 != instruction3
    
if __name__ == "__main__":
    test_instruction_equality()

    for spec in HardwareSpec:
        print(spec)
        NoiseModel(spec)