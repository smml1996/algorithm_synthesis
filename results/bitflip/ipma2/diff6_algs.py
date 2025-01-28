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
	'''fake_tenerife-0,fake_perth-1,fake_perth-4,fake_oslo-0,fake_oslo-1,fake_oslo-3,fake_kolkata-3,fake_washington-1'''
	instruction_to_ibm(qc, basis_gates, CX)
	instruction_to_ibm(qc, basis_gates, P2)
	with qc.if_test((cbits[2], 0)) as else0_:
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				pass
			with else2_:
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					instruction_to_ibm(qc, basis_gates, CX)
		with else1_:
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					instruction_to_ibm(qc, basis_gates, CX)
			with else2_:
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, CX)
				with else3_:
					instruction_to_ibm(qc, basis_gates, X0)
	with else0_:
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					instruction_to_ibm(qc, basis_gates, CX)
			with else2_:
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, CX)
				with else3_:
					instruction_to_ibm(qc, basis_gates, X0)
		with else1_:
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, CX)
				with else3_:
					instruction_to_ibm(qc, basis_gates, X0)
			with else2_:
				instruction_to_ibm(qc, basis_gates, X0)


def algorithm1(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_tenerife-1,fake_perth-2,fake_oslo-2,fake_cairo-0,fake_geneva-1,fake_washington-0'''
	instruction_to_ibm(qc, basis_gates, X0)
	instruction_to_ibm(qc, basis_gates, CX)
	instruction_to_ibm(qc, basis_gates, P2)
	with qc.if_test((cbits[2], 0)) as else0_:
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			pass
		with else1_:
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				pass
			with else2_:
				instruction_to_ibm(qc, basis_gates, X0)
	with else0_:
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				pass
			with else2_:
				instruction_to_ibm(qc, basis_gates, X0)
		with else1_:
			instruction_to_ibm(qc, basis_gates, X0)


def algorithm2(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_athens-0,fake_athens-1,fake_athens-2,fake_belem-0,fake_belem-1,fake_belem-2,fake_bogota-0,fake_bogota-1,fake_bogota-2,fake_burlington-0,fake_burlington-1,fake_burlington-2,fake_essex-0,fake_essex-1,fake_essex-2,fake_lima-0,fake_lima-1,fake_lima-2,fake_london-0,fake_london-1,fake_london-2,fake_manila-0,fake_manila-1,fake_manila-2,fake_ourense-0,fake_ourense-1,fake_ourense-2,fake_quito-0,fake_quito-1,fake_quito-2,fake_rome-0,fake_rome-1,fake_rome-2,fake_santiago-0,fake_santiago-1,fake_santiago-2,fake_vigo-0,fake_vigo-1,fake_vigo-2,fake_yorktown-0,fake_yorktown-1,fake_yorktown-2,fake_yorktown-3,fake_yorktown-4,fake_yorktown-5,fake_johannesburg-1,fake_johannesburg-2,fake_johannesburg-3,fake_johannesburg-4,fake_almaden-0,fake_almaden-1,fake_almaden-2,fake_almaden-3,fake_almaden-6,fake_almaden-7,fake_almaden-8,fake_boeblingen-0,fake_boeblingen-1,fake_boeblingen-2,fake_boeblingen-3,fake_boeblingen-4,fake_boeblingen-5,fake_boeblingen-6,fake_poughkeepsie-0,fake_poughkeepsie-1,fake_poughkeepsie-2,fake_poughkeepsie-3,fake_poughkeepsie-4,fake_poughkeepsie-5,fake_singapore-2,fake_singapore-3,fake_singapore-4,fake_tokyo-0,fake_tokyo-1,fake_tokyo-2,fake_tokyo-3,fake_tokyo-4,fake_tokyo-5,fake_tokyo-8,fake_tokyo-9,fake_lagos-0,fake_lagos-1,fake_lagos-2,fake_lagos-3,fake_lagos-4,fake_nairobi-0,fake_nairobi-1,fake_nairobi-2,fake_nairobi-3,fake_nairobi-4,fake_casablanca-0,fake_casablanca-1,fake_casablanca-2,fake_casablanca-3,fake_casablanca-4,fake_jakarta-0,fake_jakarta-1,fake_jakarta-2,fake_jakarta-3,fake_jakarta-4,fake_hanoi-2,fake_hanoi-3,fake_hanoi-4,fake_hanoi-5,fake_hanoi-6,fake_cairo-1,fake_cairo-2,fake_cairo-3,fake_cairo-4,fake_cairo-5,fake_cairo-6,fake_cairo-8,fake_cairo-9,fake_mumbai-0,fake_mumbai-1,fake_mumbai-6,fake_kolkata-2,fake_kolkata-5,fake_kolkata-6,fake_montreal-1,fake_montreal-2,fake_paris-1,fake_paris-2,fake_paris-3,fake_paris-5,fake_paris-6,fake_sydney-0,fake_sydney-1,fake_sydney-2,fake_sydney-3,fake_sydney-4,fake_sydney-5,fake_toronto-0,fake_toronto-1,fake_toronto-2,fake_toronto-3,fake_toronto-5,fake_toronto-6,fake_toronto-7,fake_toronto-8,fake_brooklyn-0,fake_brooklyn-1,fake_brooklyn-2,fake_brooklyn-3,fake_brooklyn-4,fake_brooklyn-5,fake_manhattan-3,fake_manhattan-6,fake_cambridge-1,fake_cambridge-3,fake_cambridge-4,fake_guadalupe-0,fake_guadalupe-1,fake_guadalupe-2,fake_guadalupe-3,fake_melbourne-0,fake_melbourne-1,fake_melbourne-2,fake_melbourne-4,fake_melbourne-5,fake_melbourne-6,fake_melbourne-7,fake_rochester-2,fake_rochester-3,fake_washington-2,fake_washington-3,fake_washington-6'''
	instruction_to_ibm(qc, basis_gates, CX)
	instruction_to_ibm(qc, basis_gates, P2)
	with qc.if_test((cbits[2], 0)) as else0_:
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				pass
			with else2_:
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					instruction_to_ibm(qc, basis_gates, X0)
		with else1_:
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					instruction_to_ibm(qc, basis_gates, X0)
			with else2_:
				instruction_to_ibm(qc, basis_gates, X0)
	with else0_:
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					instruction_to_ibm(qc, basis_gates, X0)
			with else2_:
				instruction_to_ibm(qc, basis_gates, X0)
		with else1_:
			instruction_to_ibm(qc, basis_gates, X0)


def algorithm3(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_johannesburg-0,fake_almaden-4,fake_almaden-5,fake_singapore-0,fake_singapore-1,fake_singapore-5,fake_tokyo-6,fake_tokyo-7,fake_perth-0,fake_perth-3,fake_hanoi-0,fake_hanoi-1,fake_cairo-7,fake_mumbai-2,fake_mumbai-3,fake_mumbai-4,fake_mumbai-5,fake_mumbai-7,fake_kolkata-0,fake_kolkata-1,fake_kolkata-4,fake_auckland-1,fake_montreal-0,fake_montreal-3,fake_paris-0,fake_paris-4,fake_toronto-4,fake_manhattan-2,fake_melbourne-3,fake_rochester-0,fake_rochester-1,fake_washington-5'''
	instruction_to_ibm(qc, basis_gates, CX)
	instruction_to_ibm(qc, basis_gates, P2)
	with qc.if_test((cbits[2], 0)) as else0_:
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			pass
		with else1_:
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				pass
			with else2_:
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					instruction_to_ibm(qc, basis_gates, X0)
	with else0_:
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				pass
			with else2_:
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					instruction_to_ibm(qc, basis_gates, X0)
		with else1_:
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					instruction_to_ibm(qc, basis_gates, X0)
			with else2_:
				instruction_to_ibm(qc, basis_gates, X0)


def algorithm4(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_oslo-4,fake_auckland-0,fake_geneva-0,fake_geneva-2'''
	instruction_to_ibm(qc, basis_gates, CX)
	instruction_to_ibm(qc, basis_gates, P2)
	with qc.if_test((cbits[2], 0)) as else0_:
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			pass
		with else1_:
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				pass
			with else2_:
				instruction_to_ibm(qc, basis_gates, X0)
	with else0_:
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				pass
			with else2_:
				instruction_to_ibm(qc, basis_gates, X0)
		with else1_:
			instruction_to_ibm(qc, basis_gates, X0)


def algorithm5(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_manhattan-0,fake_cambridge-0'''
	instruction_to_ibm(qc, basis_gates, CX)
	instruction_to_ibm(qc, basis_gates, P2)
	with qc.if_test((cbits[2], 0)) as else0_:
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, X0)
			with else2_:
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, X0)
				with else3_:
					pass
		with else1_:
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, X0)
				with else3_:
					pass
			with else2_:
				pass
	with else0_:
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, X0)
				with else3_:
					pass
			with else2_:
				pass
		with else1_:
			pass


def algorithm6(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_manhattan-1,fake_washington-4'''
	instruction_to_ibm(qc, basis_gates, CX)
	instruction_to_ibm(qc, basis_gates, P2)
	with qc.if_test((cbits[2], 0)) as else0_:
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, X0)
		with else1_:
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, X0)
			with else2_:
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, X0)
				with else3_:
					pass
	with else0_:
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, X0)
			with else2_:
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, X0)
				with else3_:
					pass
		with else1_:
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, X0)
				with else3_:
					pass
			with else2_:
				pass


def algorithm7(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_manhattan-4,fake_manhattan-5,fake_rochester-4'''
	instruction_to_ibm(qc, basis_gates, CX)
	instruction_to_ibm(qc, basis_gates, P2)
	with qc.if_test((cbits[2], 0)) as else0_:
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					instruction_to_ibm(qc, basis_gates, X0)
			with else2_:
				instruction_to_ibm(qc, basis_gates, X0)
		with else1_:
			instruction_to_ibm(qc, basis_gates, X0)
	with else0_:
		instruction_to_ibm(qc, basis_gates, X0)


def algorithm8(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_cambridge-2'''
	instruction_to_ibm(qc, basis_gates, CX)
	instruction_to_ibm(qc, basis_gates, P2)
	with qc.if_test((cbits[2], 0)) as else0_:
		pass
	with else0_:
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			pass
		with else1_:
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				pass
			with else2_:
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					instruction_to_ibm(qc, basis_gates, X0)


algorithms = []
algorithms.append(algorithm0)
algorithms.append(algorithm1)
algorithms.append(algorithm2)
algorithms.append(algorithm3)
algorithms.append(algorithm4)
algorithms.append(algorithm5)
algorithms.append(algorithm6)
algorithms.append(algorithm7)
algorithms.append(algorithm8)
