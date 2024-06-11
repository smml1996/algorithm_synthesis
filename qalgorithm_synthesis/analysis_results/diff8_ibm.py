import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qiskit import QuantumCircuit, ClassicalRegister
from utils import Instruction, instruction_to_ibm

def algorithm0(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''athens-0,athens-1,athens-2,tenerife-0,lima-0,lima-1,lima-2,rome-0,rome-1,rome-2,manila-0,manila-1,manila-2,santiago-0,santiago-1,santiago-2,bogota-0,bogota-1,bogota-2,ourense-0,ourense-1,ourense-2,yorktown-0,yorktown-1,yorktown-2,yorktown-3,yorktown-4,essex-0,essex-1,essex-2,vigo-0,vigo-1,vigo-2,burlington-0,burlington-1,burlington-2,jakarta-0,jakarta-1,jakarta-2,jakarta-3,jakarta-4,oslo-0,oslo-1,oslo-2,oslo-3,oslo-4,perth-0,perth-1,perth-2,perth-3,perth-4,lagos-0,lagos-1,lagos-2,lagos-3,lagos-4,nairobi-0,nairobi-1,nairobi-2,nairobi-3,casablanca-0,casablanca-1,casablanca-2,casablanca-3,casablanca-4,melbourne-4,melbourne-5,guadalupe-0,guadalupe-1,tokyo-0,tokyo-1,tokyo-5,tokyo-7,poughkeepsie-1,poughkeepsie-2,poughkeepsie-3,johannesburg-2,johannesburg-3,johannesburg-4,boeblingen-2,boeblingen-3,boeblingen-5,almaden-2,almaden-3,almaden-6,singapore-2,singapore-5,mumbai-0,mumbai-1,mumbai-2,mumbai-3,mumbai-4,mumbai-5,mumbai-6,mumbai-7,paris-1,paris-5,paris-6,auckland-0,kolkata-0,kolkata-1,kolkata-3,kolkata-4,kolkata-5,kolkata-6,toronto-2,toronto-3,toronto-5,toronto-6,toronto-7,toronto-8,montreal-0,montreal-2,montreal-3,sydney-3,sydney-5,cairo-0,cairo-1,cairo-5,cairo-6,cairo-7,cairo-8,cairo-9,hanoi-0,hanoi-1,hanoi-2,hanoi-3,hanoi-4,hanoi-5,geneva-0,geneva-1,cambridge-1,cambridge-3,cambridge-4,rochester-1,rochester-2,rochester-3,brooklyn-1,brooklyn-2,brooklyn-3,brooklyn-4,brooklyn-5,manhattan-3,manhattan-6,washington-3,washington-5,washington-6'''
	instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 2, 1)
	instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 2, 0)
	instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
	with qc.if_test((cbits[2], 0)) as else0_:
		instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				pass
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
		with else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
	with else0_:
		instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
		with else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)


def algorithm1(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''belem-0,belem-1,belem-2,yorktown-5,quito-0,quito-1,quito-2,london-0,london-1,london-2,melbourne-0,melbourne-1,melbourne-2,melbourne-6,melbourne-7,guadalupe-3,tokyo-2,tokyo-3,tokyo-9,poughkeepsie-0,poughkeepsie-4,poughkeepsie-5,johannesburg-1,boeblingen-1,boeblingen-4,almaden-1,almaden-7,almaden-8,singapore-4,paris-2,paris-3,toronto-0,toronto-1,sydney-0,sydney-1,rochester-4,brooklyn-0'''
	instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 2, 1)
	instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 2, 0)
	instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
	with qc.if_test((cbits[2], 0)) as else0_:
		instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
		with else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
	with else0_:
		instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
		with else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)


def algorithm2(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''tenerife-1'''
	instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 2, 1)
	instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 2, 0)
	instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
	with qc.if_test((cbits[2], 0)) as else0_:
		instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				pass
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
		with else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
	with else0_:
		instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
		with else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)


def algorithm3(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''nairobi-4,guadalupe-2,tokyo-4,tokyo-6,boeblingen-6,auckland-1,kolkata-2,toronto-4,montreal-1,sydney-2,sydney-4,cairo-2,cairo-3,cairo-4,hanoi-6,geneva-2,rochester-0,washington-0,washington-1,washington-2'''
	instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 2, 0)
	instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 2, 1)
	instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
	with qc.if_test((cbits[2], 0)) as else0_:
		instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				pass
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
		with else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
	with else0_:
		instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
		with else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)


def algorithm4(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''melbourne-3,johannesburg-0,almaden-4,almaden-5,singapore-0,singapore-1,paris-0,cambridge-2'''
	instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 2, 0)
	instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 2, 1)
	instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
	with qc.if_test((cbits[2], 0)) as else0_:
		instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
		with qc.if_test((cbits[2], 0)) as else1_:
			pass
		with else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				pass
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
	with else0_:
		instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				pass
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
		with else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)


def algorithm5(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''tokyo-8,boeblingen-0,singapore-3'''
	instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 2, 0)
	instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 2, 1)
	instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
	with qc.if_test((cbits[2], 0)) as else0_:
		instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
		with else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
	with else0_:
		instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
		with else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)


def algorithm6(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''almaden-0'''
	instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 2, 1)
	instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 2, 0)
	instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
	with qc.if_test((cbits[2], 0)) as else0_:
		instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
		with else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
	with else0_:
		instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
		with else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)


def algorithm7(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''paris-4,manhattan-2'''
	instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 2, 1)
	instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 2, 0)
	instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
	with qc.if_test((cbits[2], 0)) as else0_:
		instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
		with qc.if_test((cbits[2], 0)) as else1_:
			pass
		with else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				pass
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
	with else0_:
		instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				pass
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
		with else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					pass
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)


def algorithm8(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''cambridge-0'''
	instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 2, 1)
	instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 2, 0)
	instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
	with qc.if_test((cbits[2], 0)) as else0_:
		instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						pass
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						pass
				with else3_:
					pass
		with else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						pass
				with else3_:
					pass
			with else2_:
				pass
	with else0_:
		instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						pass
				with else3_:
					pass
			with else2_:
				pass
		with else1_:
			pass


def algorithm9(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''manhattan-0'''
	instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 2, 1)
	instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 2, 0)
	instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
	with qc.if_test((cbits[2], 0)) as else0_:
		instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						pass
		with else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						pass
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						pass
				with else3_:
					pass
	with else0_:
		instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						pass
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						pass
				with else3_:
					pass
		with else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						pass
				with else3_:
					pass
			with else2_:
				pass


def algorithm10(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''manhattan-1'''
	instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 2, 1)
	instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 2, 0)
	instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
	with qc.if_test((cbits[2], 0)) as else0_:
		instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
		with else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						pass
	with else0_:
		instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						pass
		with else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						pass
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						pass
				with else3_:
					pass


def algorithm11(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''manhattan-4,manhattan-5'''
	instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 2, 1)
	instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 2, 0)
	instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
	with qc.if_test((cbits[2], 0)) as else0_:
		instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						pass
					with else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
		with else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
	with else0_:
		instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)


def algorithm12(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''washington-4'''
	instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 2, 0)
	instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 2, 1)
	instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
	with qc.if_test((cbits[2], 0)) as else0_:
		instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						pass
		with else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						pass
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						pass
				with else3_:
					pass
	with else0_:
		instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
		with qc.if_test((cbits[2], 0)) as else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
				with else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						pass
			with else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						pass
				with else3_:
					pass
		with else1_:
			instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
			with qc.if_test((cbits[2], 0)) as else2_:
				instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
				with qc.if_test((cbits[2], 0)) as else3_:
					instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
					with qc.if_test((cbits[2], 0)) as else4_:
						instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
					with else4_:
						pass
				with else3_:
					pass
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
algorithms.append(algorithm10)
algorithms.append(algorithm11)
algorithms.append(algorithm12)
