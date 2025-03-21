import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qiskit import QuantumCircuit, ClassicalRegister
from ibm_noise_models import Instruction, instruction_to_ibm, Op

###### ACTIONS ######
CX = [Instruction(2, Op.CNOT, 0, None), Instruction(2, Op.CNOT, 1, None)]
P2 = [Instruction(2, Op.MEAS, None, None)]
X0 = [Instruction(0, Op.X, None, None)]
halt = []
###### END ACTIONS ######

def algorithm0(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	instruction_to_ibm(qc, basis_gates, CX)
	instruction_to_ibm(qc, basis_gates, P2)
	with qc.if_test((cbits, 0)):
		instruction_to_ibm(qc, basis_gates, halt)
	with qc.if_test((cbits, 4)):
		instruction_to_ibm(qc, basis_gates, X0)
		instruction_to_ibm(qc, basis_gates, halt)


algorithms = []
algorithms.append(algorithm0)
