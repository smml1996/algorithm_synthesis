import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qiskit import QuantumCircuit, ClassicalRegister
from ibm_noise_models import Instruction, instruction_to_ibm, Op

###### ACTIONS ######
RY0 = [Instruction(13, Op.SX, None, None), Instruction(13, Op.RZ, None, [3.9269908169872414]), Instruction(13, Op.SX, None, None), Instruction(13, Op.RZ, None, [9.42477796076938]), Instruction(2, Op.WRITE1, None, None)]
H1 = [Instruction(12, Op.RZ, None, [1.5707963267948966]), Instruction(12, Op.SX, None, None), Instruction(12, Op.RZ, None, [1.5707963267948966])]
X1 = [Instruction(12, Op.X, None, None)]
CX01 = [Instruction(3, Op.WRITE1, None, None), Instruction(12, Op.CNOT, 13, None)]
CX10 = [Instruction(3, Op.WRITE1, None, None), Instruction(13, Op.CNOT, 12, None)]
MEAS0 = [Instruction(13, Op.MEAS, None, None), Instruction(12, Op.MEAS, None, None), Instruction(4, Op.WRITE1, None, None)]
IS0 = [Instruction(0, Op.WRITE0, None, None), Instruction(5, Op.WRITE1, None, None)]
ISPlus = [Instruction(0, Op.WRITE1, None, None), Instruction(5, Op.WRITE1, None, None)]
halt = []
###### END ACTIONS ######

def algorithm0(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_algiers-0,fake_algiers-3,fake_hanoi-0,fake_hanoi-1,fake_hanoi-3,fake_hanoi-5,fake_cairo-0,fake_cairo-1,fake_cairo-4,fake_cairo-6,fake_cairo-7,fake_mumbai-0,fake_mumbai-1,fake_kolkata-4,fake_auckland-0,fake_auckland-1,fake_auckland-2,fake_geneva-2,fake_geneva-4,fake_montreal-0,fake_montreal-2,fake_montreal-5,fake_paris-3,fake_paris-4,fake_paris-5,fake_sydney-3,fake_sydney-4,fake_sydney-5,fake_toronto-3,fake_toronto-4,fake_valencia-3,fake_athens-2,fake_belem-0,fake_belem-1,fake_belem-2,fake_belem-3,fake_bogota-0,fake_bogota-1,fake_bogota-3,fake_essex-1,fake_lima-1,fake_london-0,fake_manila-0,fake_manila-1,fake_quito-0,fake_quito-1,fake_santiago-2,fake_santiago-3,fake_vigo-1,fake_vigo-2,fake_johannesburg-0,fake_johannesburg-1,fake_johannesburg-3,fake_almaden-0,fake_almaden-1,fake_almaden-2,fake_boeblingen-0,fake_boeblingen-2,fake_poughkeepsie-3,fake_singapore-3,fake_lagos-2,fake_nairobi-0,fake_casablanca-2,fake_oslo-0,fake_oslo-1,fake_jakarta-1,fake_jakarta-2,fake_jakarta-3,fake_brooklyn-1,fake_brooklyn-3,fake_manhattan-1,fake_manhattan-2,fake_manhattan-4,fake_manhattan-5,fake_cambridge-3,fake_cambridge-5,fake_guadalupe-0,fake_guadalupe-4,fake_melbourne-0,fake_melbourne-4,fake_melbourne-5,fake_rochester-2,fake_washington-0,fake_washington-3,fake_washington-4,fake_washington-5,fake_rochester-5,fake_algiers-2,fake_algiers-4,fake_algiers-5,fake_hanoi-4,fake_cairo-2,fake_cairo-3,fake_cairo-5,fake_cairo-8,fake_cairo-9,fake_cairo-10,fake_cairo-11,fake_mumbai-4,fake_kolkata-0,fake_kolkata-2,fake_auckland-4,fake_geneva-0,fake_montreal-3,fake_montreal-4,fake_paris-1,fake_sydney-2,fake_toronto-0,fake_toronto-2,fake_toronto-5,fake_valencia-0,fake_valencia-1,fake_athens-0,fake_essex-0,fake_essex-3,fake_lima-2,fake_london-2,fake_ourense-1,fake_quito-2,fake_rome-0,fake_rome-1,fake_rome-2,fake_santiago-0,fake_santiago-1,fake_yorktown-1,fake_almaden-4,fake_boeblingen-4,fake_poughkeepsie-0,fake_poughkeepsie-2,fake_singapore-1,fake_singapore-4,fake_perth-1,fake_perth-3,fake_nairobi-1,fake_nairobi-2,fake_casablanca-0,fake_casablanca-1,fake_oslo-2,fake_jakarta-0,fake_brooklyn-0,fake_brooklyn-2,fake_brooklyn-4,fake_manhattan-0,fake_cambridge-0,fake_cambridge-1,fake_guadalupe-2,fake_guadalupe-5,fake_melbourne-2,fake_melbourne-3,fake_rochester-1,fake_rochester-4,fake_washington-1'''
	instruction_to_ibm(qc, basis_gates, RY0)
	instruction_to_ibm(qc, basis_gates, MEAS0)
	with qc.if_test((cbits, 20)):
		instruction_to_ibm(qc, basis_gates, halt)
	with qc.if_test((cbits, 21)):
		instruction_to_ibm(qc, basis_gates, halt)
	with qc.if_test((cbits, 22)):
		instruction_to_ibm(qc, basis_gates, halt)
	with qc.if_test((cbits, 23)):
		instruction_to_ibm(qc, basis_gates, halt)


def algorithm1(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_algiers-1,fake_hanoi-2,fake_hanoi-6,fake_mumbai-2,fake_mumbai-3,fake_kolkata-1,fake_kolkata-3,fake_auckland-3,fake_geneva-1,fake_geneva-3,fake_montreal-1,fake_paris-0,fake_paris-2,fake_sydney-0,fake_sydney-1,fake_toronto-1,fake_valencia-2,fake_athens-1,fake_bogota-2,fake_burlington-0,fake_burlington-1,fake_burlington-2,fake_burlington-3,fake_essex-2,fake_lima-0,fake_london-1,fake_manila-2,fake_ourense-0,fake_ourense-2,fake_vigo-0,fake_vigo-3,fake_yorktown-0,fake_yorktown-2,fake_johannesburg-2,fake_johannesburg-4,fake_johannesburg-5,fake_almaden-3,fake_boeblingen-1,fake_boeblingen-3,fake_poughkeepsie-1,fake_poughkeepsie-4,fake_singapore-0,fake_singapore-2,fake_singapore-5,fake_perth-0,fake_perth-2,fake_lagos-0,fake_lagos-1,fake_nairobi-3,fake_casablanca-3,fake_brooklyn-5,fake_manhattan-3,fake_cambridge-2,fake_cambridge-4,fake_guadalupe-1,fake_guadalupe-3,fake_melbourne-1,fake_rochester-0,fake_rochester-3,fake_washington-2'''
	instruction_to_ibm(qc, basis_gates, RY0)
	instruction_to_ibm(qc, basis_gates, CX01)
	instruction_to_ibm(qc, basis_gates, MEAS0)
	with qc.if_test((cbits, 28)):
		instruction_to_ibm(qc, basis_gates, halt)
	with qc.if_test((cbits, 29)):
		instruction_to_ibm(qc, basis_gates, IS0)
		instruction_to_ibm(qc, basis_gates, halt)
	with qc.if_test((cbits, 30)):
		instruction_to_ibm(qc, basis_gates, ISPlus)
		instruction_to_ibm(qc, basis_gates, halt)
	with qc.if_test((cbits, 31)):
		instruction_to_ibm(qc, basis_gates, halt)


algorithms = []
algorithms.append(algorithm0)
algorithms.append(algorithm1)
