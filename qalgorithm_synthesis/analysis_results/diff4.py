import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *

algorithms = [{'instruction': Instruction.CNOT, 'target': 2, 'control': 1, 'params': None, 'next': {'instruction': Instruction.CNOT, 'target': 2, 'control': 0, 'params': None, 'next': {'instruction': Instruction.MEAS, 'target': 2, 'control': None, 'params': None, 'next': None, 'case0': None, 'case1': {'instruction': Instruction.X, 'target': 0, 'control': None, 'params': None, 'next': None, 'case0': None, 'case1': None, 'count_meas': 0, 'depth': 0}, 'count_meas': 0, 'depth': 0}, 'case0': None, 'case1': None, 'count_meas': 0, 'depth': 0}, 'case0': None, 'case1': None, 'count_meas': 0, 'depth': 0} 
,{'instruction': Instruction.CNOT, 'target': 2, 'control': 1, 'params': None, 'next': {'instruction': Instruction.CNOT, 'target': 2, 'control': 0, 'params': None, 'next': {'instruction': Instruction.MEAS, 'target': 2, 'control': None, 'params': None, 'next': None, 'case0': {'instruction': Instruction.X, 'target': 0, 'control': None, 'params': None, 'next': None, 'case0': None, 'case1': None, 'count_meas': 0, 'depth': 0}, 'case1': None, 'count_meas': 0, 'depth': 0}, 'case0': None, 'case1': None, 'count_meas': 0, 'depth': 0}, 'case0': None, 'case1': None, 'count_meas': 0, 'depth': 0} 
]