import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qiskit import QuantumCircuit, ClassicalRegister
from ibm_noise_models import Instruction, instruction_to_ibm, Op

###### ACTIONS ######
H0 = [Instruction(13, Op.RZ, None, [1.5707963267948966]), Instruction(13, Op.SX, None, None), Instruction(13, Op.RZ, None, [1.5707963267948966])]
H1 = [Instruction(12, Op.RZ, None, [1.5707963267948966]), Instruction(12, Op.SX, None, None), Instruction(12, Op.RZ, None, [1.5707963267948966])]
RY0 = [Instruction(13, Op.SX, None, None), Instruction(13, Op.RZ, None, [3.9269908169872414]), Instruction(13, Op.SX, None, None), Instruction(13, Op.RZ, None, [9.42477796076938])]
RY1 = [Instruction(12, Op.SX, None, None), Instruction(12, Op.RZ, None, [3.9269908169872414]), Instruction(12, Op.SX, None, None), Instruction(12, Op.RZ, None, [9.42477796076938])]
rycx01 = [Instruction(12, Op.SX, None, None), Instruction(12, Op.RZ, None, [3.9269908169872414]), Instruction(12, Op.SX, None, None), Instruction(12, Op.RZ, None, [9.42477796076938]), Instruction(12, Op.CNOT, 13, None), Instruction(4, Op.WRITE1, None, None)]
hcx10 = [Instruction(12, Op.RZ, None, [1.5707963267948966]), Instruction(12, Op.SX, None, None), Instruction(12, Op.RZ, None, [1.5707963267948966]), Instruction(13, Op.CNOT, 12, None), Instruction(4, Op.WRITE1, None, None)]
MEAS = [Instruction(13, Op.MEAS, None, None), Instruction(12, Op.MEAS, None, None), Instruction(2, Op.WRITE1, None, None)]
IS0 = [Instruction(0, Op.WRITE0, None, None), Instruction(3, Op.WRITE1, None, None)]
ISPlus = [Instruction(0, Op.WRITE1, None, None), Instruction(3, Op.WRITE1, None, None)]
halt = []
###### END ACTIONS ######

def algorithm0(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_algiers-0,fake_algiers-1,fake_hanoi-3,fake_hanoi-6,fake_cairo-9,fake_kolkata-3,fake_auckland-0,fake_auckland-1,fake_montreal-0,fake_paris-2,fake_paris-4,fake_sydney-2,fake_sydney-5,fake_toronto-2,fake_toronto-3,fake_toronto-5,fake_athens-1,fake_athens-2,fake_bogota-1,fake_bogota-2,fake_manila-0,fake_manila-1,fake_ourense-1,fake_quito-1,fake_santiago-0,fake_santiago-3,fake_johannesburg-0,fake_poughkeepsie-1,fake_poughkeepsie-3,fake_poughkeepsie-4,fake_singapore-1,fake_singapore-3,fake_perth-0,fake_perth-2,fake_nairobi-3,fake_casablanca-1,fake_casablanca-3,fake_oslo-0,fake_manhattan-3,fake_melbourne-1,fake_rochester-2,fake_washington-1,fake_washington-4,fake_washington-5'''
	instruction_to_ibm(qc, basis_gates, RY0)
	instruction_to_ibm(qc, basis_gates, MEAS)
	with qc.if_test((cbits, 4)):
		instruction_to_ibm(qc, basis_gates, halt)
	with qc.if_test((cbits, 5)):
		instruction_to_ibm(qc, basis_gates, halt)
	with qc.if_test((cbits, 6)):
		instruction_to_ibm(qc, basis_gates, halt)
	with qc.if_test((cbits, 7)):
		instruction_to_ibm(qc, basis_gates, halt)


def algorithm7(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_geneva-3,fake_london-1,fake_vigo-0,fake_yorktown-0,fake_johannesburg-2,fake_singapore-0,fake_singapore-5,fake_lagos-1,fake_cambridge-4,fake_rochester-3'''
	instruction_to_ibm(qc, basis_gates, RY0)
	instruction_to_ibm(qc, basis_gates, rycx01)
	instruction_to_ibm(qc, basis_gates, RY1)
	instruction_to_ibm(qc, basis_gates, MEAS)
	with qc.if_test((cbits, 20)):
		instruction_to_ibm(qc, basis_gates, halt)
	with qc.if_test((cbits, 21)):
		instruction_to_ibm(qc, basis_gates, IS0)
		instruction_to_ibm(qc, basis_gates, halt)
	with qc.if_test((cbits, 22)):
		instruction_to_ibm(qc, basis_gates, halt)
	with qc.if_test((cbits, 23)):
		instruction_to_ibm(qc, basis_gates, halt)



def algorithm10(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_yorktown-2'''
	instruction_to_ibm(qc, basis_gates, hcx10)
	instruction_to_ibm(qc, basis_gates, H1)
	instruction_to_ibm(qc, basis_gates, H0)
	instruction_to_ibm(qc, basis_gates, MEAS)
	with qc.if_test((cbits, 20)):
		instruction_to_ibm(qc, basis_gates, ISPlus)
		instruction_to_ibm(qc, basis_gates, halt)
	with qc.if_test((cbits, 21)):
		instruction_to_ibm(qc, basis_gates, halt)
	with qc.if_test((cbits, 22)):
		instruction_to_ibm(qc, basis_gates, halt)
	with qc.if_test((cbits, 23)):
		instruction_to_ibm(qc, basis_gates, IS0)
		instruction_to_ibm(qc, basis_gates, halt)


def algorithm12(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_lagos-0'''
	instruction_to_ibm(qc, basis_gates, hcx10)
	instruction_to_ibm(qc, basis_gates, RY0)
	instruction_to_ibm(qc, basis_gates, H1)
	instruction_to_ibm(qc, basis_gates, MEAS)
	with qc.if_test((cbits, 20)):
		instruction_to_ibm(qc, basis_gates, ISPlus)
		instruction_to_ibm(qc, basis_gates, halt)
	with qc.if_test((cbits, 21)):
		instruction_to_ibm(qc, basis_gates, halt)
	with qc.if_test((cbits, 22)):
		instruction_to_ibm(qc, basis_gates, halt)
	with qc.if_test((cbits, 23)):
		instruction_to_ibm(qc, basis_gates, IS0)
		instruction_to_ibm(qc, basis_gates, halt)


def algorithm13(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_rochester-0'''
	instruction_to_ibm(qc, basis_gates, hcx10)
	instruction_to_ibm(qc, basis_gates, H1)
	instruction_to_ibm(qc, basis_gates, MEAS)
	with qc.if_test((cbits, 20)):
		instruction_to_ibm(qc, basis_gates, ISPlus)
		instruction_to_ibm(qc, basis_gates, halt)
	with qc.if_test((cbits, 21)):
		instruction_to_ibm(qc, basis_gates, halt)
	with qc.if_test((cbits, 22)):
		instruction_to_ibm(qc, basis_gates, halt)
	with qc.if_test((cbits, 23)):
		instruction_to_ibm(qc, basis_gates, IS0)
		instruction_to_ibm(qc, basis_gates, halt)


algorithms = []
algorithms.append(algorithm0)
algorithms.append(algorithm7)
algorithms.append(algorithm10)
algorithms.append(algorithm12)
algorithms.append(algorithm13)
