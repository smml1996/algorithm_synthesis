import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qiskit import QuantumCircuit, ClassicalRegister
from ibm_noise_models import Instruction, instruction_to_ibm, Op

###### ACTIONS ######
P2 = [Instruction(2, Op.MEAS, None, None)]
Z0 = [Instruction(0, Op.SX, None, None), Instruction(0, Op.RZ, None, [3.141592653589793]), Instruction(0, Op.SX, None, None)]
H = [Instruction(2, Op.RZ, None, [1.5707963267948966]), Instruction(2, Op.SX, None, None), Instruction(2, Op.RZ, None, [1.5707963267948966]), Instruction(1, Op.CNOT, 2, None), Instruction(0, Op.CNOT, 2, None), Instruction(2, Op.RZ, None, [1.5707963267948966]), Instruction(2, Op.SX, None, None), Instruction(2, Op.RZ, None, [1.5707963267948966])]
###### END ACTIONS ######

def algorithm0(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_johannesburg-0,fake_almaden-3,fake_singapore-0,fake_paris-1,fake_paris-2,fake_cambridge-2,fake_melbourne-3'''
	H
	instruction_to_ibm(qc, basis_gates, H)
	P2
	instruction_to_ibm(qc, basis_gates, P2)
	with qc.if_test((cbits[2], 0)) as else0_:
		P2
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			pass
		with else1_:
			P2
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				pass
			with else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
	with else0_:
		P2
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			P2
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				pass
			with else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
		with else1_:
			P2
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
			with else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
				with else3_:
					Z0
					instruction_to_ibm(qc, basis_gates, Z0)


def algorithm1(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_johannesburg-1,fake_almaden-0,fake_boeblingen-1,fake_poughkeepsie-0,fake_poughkeepsie-3,fake_singapore-2,fake_tokyo-2,fake_brooklyn-0,fake_melbourne-2,fake_rochester-4'''
	H
	instruction_to_ibm(qc, basis_gates, H)
	P2
	instruction_to_ibm(qc, basis_gates, P2)
	with qc.if_test((cbits[2], 0)) as else0_:
		P2
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			P2
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
			with else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
				with else3_:
					Z0
					instruction_to_ibm(qc, basis_gates, Z0)
		with else1_:
			P2
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
				with else3_:
					Z0
					instruction_to_ibm(qc, basis_gates, Z0)
			with else2_:
				Z0
				instruction_to_ibm(qc, basis_gates, Z0)
	with else0_:
		P2
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			P2
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
				with else3_:
					Z0
					instruction_to_ibm(qc, basis_gates, Z0)
			with else2_:
				Z0
				instruction_to_ibm(qc, basis_gates, Z0)
		with else1_:
			Z0
			instruction_to_ibm(qc, basis_gates, Z0)


def algorithm2(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_johannesburg-2,fake_almaden-1,fake_almaden-2,fake_boeblingen-0,fake_boeblingen-2,fake_boeblingen-3,fake_poughkeepsie-1,fake_poughkeepsie-2,fake_singapore-3,fake_tokyo-0,fake_tokyo-1,fake_tokyo-3,fake_tokyo-4,fake_hanoi-0,fake_hanoi-1,fake_hanoi-2,fake_hanoi-3,fake_cairo-0,fake_cairo-1,fake_cairo-2,fake_cairo-3,fake_mumbai-0,fake_mumbai-1,fake_mumbai-2,fake_mumbai-3,fake_kolkata-0,fake_kolkata-1,fake_kolkata-2,fake_kolkata-3,fake_auckland-0,fake_auckland-1,fake_geneva-1,fake_montreal-0,fake_montreal-1,fake_montreal-2,fake_paris-0,fake_sydney-0,fake_sydney-1,fake_toronto-2,fake_brooklyn-1,fake_brooklyn-2,fake_brooklyn-3,fake_manhattan-2,fake_cambridge-1,fake_guadalupe-0,fake_guadalupe-1,fake_melbourne-1,fake_rochester-0,fake_rochester-1,fake_rochester-2,fake_washington-0'''
	H
	instruction_to_ibm(qc, basis_gates, H)
	P2
	instruction_to_ibm(qc, basis_gates, P2)
	with qc.if_test((cbits[2], 0)) as else0_:
		P2
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			P2
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				pass
			with else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
		with else1_:
			P2
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
			with else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
				with else3_:
					Z0
					instruction_to_ibm(qc, basis_gates, Z0)
	with else0_:
		P2
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			P2
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
			with else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
				with else3_:
					Z0
					instruction_to_ibm(qc, basis_gates, Z0)
		with else1_:
			P2
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
				with else3_:
					Z0
					instruction_to_ibm(qc, basis_gates, Z0)
			with else2_:
				Z0
				instruction_to_ibm(qc, basis_gates, Z0)


def algorithm3(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_johannesburg-3,fake_singapore-1,fake_manhattan-4,fake_cambridge-3,fake_rochester-3'''
	H
	instruction_to_ibm(qc, basis_gates, H)
	P2
	instruction_to_ibm(qc, basis_gates, P2)
	with qc.if_test((cbits[2], 0)) as else0_:
		P2
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			P2
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				pass
			with else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
		with else1_:
			P2
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
			with else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
				with else3_:
					Z0
					instruction_to_ibm(qc, basis_gates, Z0)
	with else0_:
		H
		instruction_to_ibm(qc, basis_gates, H)
		P2
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			P2
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				Z0
				instruction_to_ibm(qc, basis_gates, Z0)
			with else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					Z0
					instruction_to_ibm(qc, basis_gates, Z0)
				with else3_:
					pass
		with else1_:
			H
			instruction_to_ibm(qc, basis_gates, H)
			P2
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				Z0
				instruction_to_ibm(qc, basis_gates, Z0)
			with else2_:
				pass


def algorithm4(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_geneva-0,fake_manhattan-0,fake_washington-1'''
	H
	instruction_to_ibm(qc, basis_gates, H)
	P2
	instruction_to_ibm(qc, basis_gates, P2)
	with qc.if_test((cbits[2], 0)) as else0_:
		P2
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			P2
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				Z0
				instruction_to_ibm(qc, basis_gates, Z0)
			with else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					Z0
					instruction_to_ibm(qc, basis_gates, Z0)
				with else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
					with else4_:
						pass
		with else1_:
			P2
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					Z0
					instruction_to_ibm(qc, basis_gates, Z0)
				with else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
					with else4_:
						pass
			with else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
					with else4_:
						pass
				with else3_:
					pass
	with else0_:
		P2
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			P2
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					Z0
					instruction_to_ibm(qc, basis_gates, Z0)
				with else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
					with else4_:
						pass
			with else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
					with else4_:
						pass
				with else3_:
					pass
		with else1_:
			P2
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
					with else4_:
						pass
				with else3_:
					pass
			with else2_:
				pass


def algorithm5(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_toronto-0,fake_toronto-1,fake_washington-2'''
	H
	instruction_to_ibm(qc, basis_gates, H)
	P2
	instruction_to_ibm(qc, basis_gates, P2)
	with qc.if_test((cbits[2], 0)) as else0_:
		H
		instruction_to_ibm(qc, basis_gates, H)
		P2
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			P2
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				pass
			with else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					Z0
					instruction_to_ibm(qc, basis_gates, Z0)
		with else1_:
			H
			instruction_to_ibm(qc, basis_gates, H)
			P2
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				Z0
				instruction_to_ibm(qc, basis_gates, Z0)
			with else2_:
				pass
	with else0_:
		H
		instruction_to_ibm(qc, basis_gates, H)
		P2
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			P2
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				Z0
				instruction_to_ibm(qc, basis_gates, Z0)
			with else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					Z0
					instruction_to_ibm(qc, basis_gates, Z0)
				with else3_:
					pass
		with else1_:
			H
			instruction_to_ibm(qc, basis_gates, H)
			P2
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				Z0
				instruction_to_ibm(qc, basis_gates, Z0)
			with else2_:
				pass


def algorithm6(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_manhattan-1'''
	H
	instruction_to_ibm(qc, basis_gates, H)
	P2
	instruction_to_ibm(qc, basis_gates, P2)
	with qc.if_test((cbits[2], 0)) as else0_:
		P2
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			Z0
			instruction_to_ibm(qc, basis_gates, Z0)
		with else1_:
			P2
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				Z0
				instruction_to_ibm(qc, basis_gates, Z0)
			with else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					Z0
					instruction_to_ibm(qc, basis_gates, Z0)
				with else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
					with else4_:
						pass
	with else0_:
		P2
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			P2
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				Z0
				instruction_to_ibm(qc, basis_gates, Z0)
			with else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					Z0
					instruction_to_ibm(qc, basis_gates, Z0)
				with else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
					with else4_:
						pass
		with else1_:
			P2
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					Z0
					instruction_to_ibm(qc, basis_gates, Z0)
				with else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
					with else4_:
						pass
			with else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
					with else4_:
						pass
				with else3_:
					pass


def algorithm7(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_manhattan-3'''
	H
	instruction_to_ibm(qc, basis_gates, H)
	P2
	instruction_to_ibm(qc, basis_gates, P2)
	with qc.if_test((cbits[2], 0)) as else0_:
		P2
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			P2
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
				with else3_:
					Z0
					instruction_to_ibm(qc, basis_gates, Z0)
			with else2_:
				Z0
				instruction_to_ibm(qc, basis_gates, Z0)
		with else1_:
			Z0
			instruction_to_ibm(qc, basis_gates, Z0)
	with else0_:
		Z0
		instruction_to_ibm(qc, basis_gates, Z0)


def algorithm8(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_cambridge-0'''
	H
	instruction_to_ibm(qc, basis_gates, H)
	P2
	instruction_to_ibm(qc, basis_gates, P2)
	with qc.if_test((cbits[2], 0)) as else0_:
		P2
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			P2
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					Z0
					instruction_to_ibm(qc, basis_gates, Z0)
				with else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
					with else4_:
						pass
			with else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
					with else4_:
						pass
				with else3_:
					pass
		with else1_:
			P2
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
					with else4_:
						pass
				with else3_:
					pass
			with else2_:
				pass
	with else0_:
		P2
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			P2
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
					with else4_:
						pass
				with else3_:
					pass
			with else2_:
				pass
		with else1_:
			pass


def algorithm9(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_melbourne-0'''
	H
	instruction_to_ibm(qc, basis_gates, H)
	P2
	instruction_to_ibm(qc, basis_gates, P2)
	with qc.if_test((cbits[2], 0)) as else0_:
		P2
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			P2
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
			with else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
				with else3_:
					Z0
					instruction_to_ibm(qc, basis_gates, Z0)
		with else1_:
			P2
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					P2
					instruction_to_ibm(qc, basis_gates, P2)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						Z0
						instruction_to_ibm(qc, basis_gates, Z0)
				with else3_:
					Z0
					instruction_to_ibm(qc, basis_gates, Z0)
			with else2_:
				Z0
				instruction_to_ibm(qc, basis_gates, Z0)
	with else0_:
		H
		instruction_to_ibm(qc, basis_gates, H)
		P2
		instruction_to_ibm(qc, basis_gates, P2)
		with qc.if_test((cbits[2], 0)) as else1_:
			P2
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				Z0
				instruction_to_ibm(qc, basis_gates, Z0)
			with else2_:
				P2
				instruction_to_ibm(qc, basis_gates, P2)
				with qc.if_test((cbits[2], 0)) as else3_:
					Z0
					instruction_to_ibm(qc, basis_gates, Z0)
				with else3_:
					pass
		with else1_:
			H
			instruction_to_ibm(qc, basis_gates, H)
			P2
			instruction_to_ibm(qc, basis_gates, P2)
			with qc.if_test((cbits[2], 0)) as else2_:
				Z0
				instruction_to_ibm(qc, basis_gates, Z0)
			with else2_:
				pass


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
algorithms.append(algorithm9)
