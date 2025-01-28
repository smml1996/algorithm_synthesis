import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qiskit import QuantumCircuit, ClassicalRegister
from ibm_noise_models import Instruction, instruction_to_ibm, Op

###### ACTIONS ######
H0 = [Instruction(124, Op.RZ, None, [1.5707963267948966]), Instruction(124, Op.SX, None, None), Instruction(124, Op.RZ, None, [1.5707963267948966])]
H1 = [Instruction(125, Op.RZ, None, [1.5707963267948966]), Instruction(125, Op.SX, None, None), Instruction(125, Op.RZ, None, [1.5707963267948966])]
H2 = [Instruction(126, Op.RZ, None, [1.5707963267948966]), Instruction(126, Op.SX, None, None), Instruction(126, Op.RZ, None, [1.5707963267948966])]
CX01 = [Instruction(125, Op.CNOT, 124, None)]
CX10 = [Instruction(124, Op.CNOT, 125, None)]
CX12 = [Instruction(126, Op.CNOT, 125, None)]
CX21 = [Instruction(125, Op.CNOT, 126, None)]
halt = []
###### END ACTIONS ######

def algorithm0(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_johannesburg-0,fake_johannesburg-2,fake_johannesburg-3,fake_johannesburg-4,fake_johannesburg-8,fake_johannesburg-19,fake_johannesburg-22,fake_johannesburg-26,fake_almaden-1,fake_almaden-2,fake_almaden-5,fake_almaden-11,fake_almaden-24,fake_almaden-29,fake_almaden-34,fake_boeblingen-0,fake_boeblingen-1,fake_boeblingen-6,fake_boeblingen-10,fake_boeblingen-13,fake_boeblingen-16,fake_boeblingen-19,fake_boeblingen-24,fake_boeblingen-32,fake_boeblingen-35,fake_poughkeepsie-0,fake_poughkeepsie-5,fake_poughkeepsie-9,fake_poughkeepsie-19,fake_poughkeepsie-25,fake_singapore-0,fake_singapore-2,fake_singapore-14,fake_singapore-15,fake_singapore-20,fake_singapore-24,fake_singapore-27,fake_singapore-33,fake_singapore-34,fake_tokyo-10,fake_tokyo-22,fake_tokyo-26,fake_tokyo-27,fake_tokyo-28,fake_tokyo-33,fake_tokyo-42,fake_tokyo-49,fake_tokyo-59,fake_tokyo-63,fake_tokyo-69,fake_tokyo-83,fake_hanoi-11,fake_hanoi-14,fake_hanoi-16,fake_hanoi-26,fake_hanoi-27,fake_hanoi-29,fake_cairo-2,fake_cairo-4,fake_cairo-9,fake_cairo-10,fake_cairo-15,fake_cairo-22,fake_cairo-31,fake_cairo-35,fake_mumbai-6,fake_mumbai-8,fake_mumbai-15,fake_mumbai-18,fake_mumbai-21,fake_mumbai-25,fake_mumbai-28,fake_kolkata-5,fake_kolkata-8,fake_kolkata-10,fake_kolkata-11,fake_kolkata-12,fake_kolkata-22,fake_kolkata-23,fake_kolkata-36,fake_auckland-9,fake_auckland-10,fake_auckland-16,fake_auckland-19,fake_auckland-27,fake_auckland-29,fake_auckland-36,fake_geneva-12,fake_geneva-14,fake_geneva-18,fake_geneva-22,fake_geneva-26,fake_geneva-35,fake_montreal-4,fake_montreal-5,fake_montreal-9,fake_montreal-10,fake_montreal-16,fake_montreal-18,fake_montreal-19,fake_montreal-22,fake_montreal-25,fake_montreal-28,fake_montreal-31,fake_paris-0,fake_paris-2,fake_paris-12,fake_paris-14,fake_paris-18,fake_paris-21,fake_paris-22,fake_paris-32,fake_paris-34,fake_sydney-0,fake_sydney-1,fake_sydney-6,fake_sydney-12,fake_sydney-18,fake_sydney-22,fake_sydney-23,fake_toronto-8,fake_toronto-9,fake_toronto-11,fake_toronto-12,fake_toronto-18,fake_toronto-22,fake_toronto-29,fake_toronto-36,fake_brooklyn-0,fake_brooklyn-2,fake_brooklyn-5,fake_brooklyn-6,fake_brooklyn-10,fake_brooklyn-13,fake_brooklyn-16,fake_brooklyn-31,fake_brooklyn-33,fake_brooklyn-35,fake_brooklyn-39,fake_brooklyn-48,fake_brooklyn-56,fake_brooklyn-65,fake_brooklyn-70,fake_brooklyn-71,fake_brooklyn-73,fake_brooklyn-74,fake_brooklyn-81,fake_brooklyn-89,fake_brooklyn-92,fake_brooklyn-94,fake_manhattan-0,fake_manhattan-3,fake_manhattan-4,fake_manhattan-11,fake_manhattan-13,fake_manhattan-18,fake_manhattan-26,fake_manhattan-30,fake_manhattan-34,fake_manhattan-39,fake_manhattan-45,fake_manhattan-46,fake_manhattan-48,fake_manhattan-54,fake_manhattan-55,fake_manhattan-59,fake_manhattan-60,fake_manhattan-66,fake_manhattan-73,fake_manhattan-80,fake_cambridge-2,fake_cambridge-11,fake_cambridge-13,fake_cambridge-21,fake_cambridge-34,fake_cambridge-35,fake_guadalupe-4,fake_guadalupe-14,fake_guadalupe-18,fake_rochester-3,fake_rochester-5,fake_rochester-11,fake_rochester-16,fake_rochester-17,fake_rochester-27,fake_rochester-29,fake_rochester-32,fake_rochester-34,fake_rochester-35,fake_rochester-38,fake_rochester-39,fake_rochester-41,fake_rochester-46,fake_rochester-51,fake_rochester-53,fake_rochester-64,fake_rochester-65,fake_rochester-66,fake_rochester-72,fake_rochester-73,fake_washington-2,fake_washington-3,fake_washington-4,fake_washington-9,fake_washington-11,fake_washington-17,fake_washington-28,fake_washington-29,fake_washington-30,fake_washington-39,fake_washington-61,fake_washington-64,fake_washington-65,fake_washington-76,fake_washington-78,fake_washington-81,fake_washington-90,fake_washington-91,fake_washington-97,fake_washington-98,fake_washington-99,fake_washington-103,fake_washington-116,fake_washington-118,fake_washington-127,fake_washington-129,fake_washington-137,fake_washington-141,fake_washington-148,fake_washington-154,fake_washington-163,fake_washington-164,fake_washington-171,fake_washington-175,fake_washington-177,fake_washington-183,fake_washington-186,fake_washington-187,fake_washington-188'''
	instruction_to_ibm(qc, basis_gates, H1)
	instruction_to_ibm(qc, basis_gates, CX12)
	instruction_to_ibm(qc, basis_gates, CX10)


def algorithm1(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_johannesburg-1,fake_johannesburg-10,fake_almaden-3,fake_almaden-8,fake_almaden-12,fake_almaden-25,fake_almaden-30,fake_boeblingen-3,fake_poughkeepsie-10,fake_tokyo-1,fake_tokyo-51,fake_tokyo-52,fake_tokyo-81,fake_hanoi-3,fake_hanoi-20,fake_hanoi-30,fake_cairo-3,fake_cairo-13,fake_kolkata-13,fake_auckland-3,fake_auckland-13,fake_auckland-20,fake_auckland-30,fake_geneva-3,fake_montreal-20,fake_sydney-3,fake_toronto-30,fake_brooklyn-8,fake_brooklyn-14,fake_brooklyn-32,fake_brooklyn-58,fake_manhattan-14,fake_manhattan-32,fake_guadalupe-19,fake_rochester-12,fake_rochester-40,fake_rochester-74,fake_washington-18,fake_washington-45,fake_washington-66,fake_washington-93,fake_washington-100,fake_washington-128,fake_washington-161'''
	instruction_to_ibm(qc, basis_gates, H0)
	instruction_to_ibm(qc, basis_gates, CX02)
	instruction_to_ibm(qc, basis_gates, CX01)


def algorithm2(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_johannesburg-5,fake_johannesburg-11,fake_johannesburg-12,fake_johannesburg-21,fake_johannesburg-25,fake_johannesburg-27,fake_almaden-6,fake_almaden-7,fake_almaden-13,fake_almaden-14,fake_almaden-15,fake_almaden-27,fake_almaden-28,fake_almaden-32,fake_almaden-33,fake_almaden-35,fake_boeblingen-5,fake_boeblingen-11,fake_boeblingen-22,fake_poughkeepsie-6,fake_poughkeepsie-8,fake_poughkeepsie-13,fake_poughkeepsie-14,fake_poughkeepsie-21,fake_poughkeepsie-26,fake_poughkeepsie-31,fake_singapore-6,fake_singapore-11,fake_singapore-13,fake_singapore-16,fake_singapore-19,fake_singapore-35,fake_tokyo-0,fake_tokyo-4,fake_tokyo-5,fake_tokyo-6,fake_tokyo-36,fake_tokyo-39,fake_tokyo-43,fake_tokyo-55,fake_tokyo-57,fake_tokyo-58,fake_tokyo-64,fake_tokyo-72,fake_tokyo-73,fake_tokyo-82,fake_hanoi-0,fake_hanoi-1,fake_hanoi-19,fake_hanoi-21,fake_hanoi-31,fake_cairo-0,fake_cairo-1,fake_cairo-5,fake_cairo-6,fake_cairo-14,fake_cairo-19,fake_cairo-25,fake_mumbai-1,fake_mumbai-9,fake_mumbai-10,fake_mumbai-19,fake_mumbai-29,fake_kolkata-1,fake_kolkata-6,fake_kolkata-15,fake_kolkata-16,fake_kolkata-19,fake_kolkata-26,fake_kolkata-27,fake_kolkata-29,fake_auckland-4,fake_auckland-11,fake_auckland-14,fake_auckland-21,fake_auckland-22,fake_auckland-31,fake_auckland-32,fake_geneva-0,fake_geneva-1,fake_geneva-4,fake_geneva-15,fake_geneva-19,fake_geneva-23,fake_geneva-32,fake_geneva-36,fake_montreal-6,fake_montreal-11,fake_montreal-14,fake_montreal-21,fake_montreal-29,fake_montreal-34,fake_montreal-36,fake_paris-5,fake_paris-11,fake_paris-15,fake_paris-16,fake_paris-23,fake_paris-25,fake_paris-35,fake_sydney-9,fake_sydney-10,fake_sydney-16,fake_sydney-28,fake_toronto-1,fake_toronto-15,fake_toronto-16,fake_toronto-25,fake_toronto-28,fake_toronto-31,fake_toronto-32,fake_brooklyn-11,fake_brooklyn-15,fake_brooklyn-20,fake_brooklyn-34,fake_brooklyn-36,fake_brooklyn-47,fake_brooklyn-50,fake_brooklyn-52,fake_brooklyn-59,fake_brooklyn-60,fake_brooklyn-66,fake_brooklyn-67,fake_brooklyn-77,fake_brooklyn-87,fake_brooklyn-90,fake_brooklyn-93,fake_manhattan-5,fake_manhattan-6,fake_manhattan-15,fake_manhattan-27,fake_manhattan-28,fake_manhattan-33,fake_manhattan-47,fake_manhattan-50,fake_manhattan-57,fake_manhattan-62,fake_manhattan-65,fake_manhattan-67,fake_manhattan-68,fake_manhattan-81,fake_manhattan-87,fake_manhattan-88,fake_manhattan-92,fake_manhattan-93,fake_cambridge-0,fake_cambridge-23,fake_cambridge-31,fake_guadalupe-1,fake_guadalupe-5,fake_guadalupe-8,fake_guadalupe-11,fake_rochester-0,fake_rochester-2,fake_rochester-4,fake_rochester-6,fake_rochester-13,fake_rochester-14,fake_rochester-25,fake_rochester-31,fake_rochester-36,fake_rochester-43,fake_rochester-49,fake_rochester-55,fake_rochester-59,fake_rochester-62,fake_washington-5,fake_washington-6,fake_washington-10,fake_washington-14,fake_washington-16,fake_washington-19,fake_washington-20,fake_washington-22,fake_washington-24,fake_washington-35,fake_washington-46,fake_washington-54,fake_washington-62,fake_washington-67,fake_washington-68,fake_washington-82,fake_washington-88,fake_washington-92,fake_washington-94,fake_washington-101,fake_washington-122,fake_washington-125,fake_washington-126,fake_washington-130,fake_washington-138,fake_washington-139,fake_washington-140,fake_washington-150,fake_washington-151,fake_washington-162,fake_washington-170,fake_washington-184,fake_washington-189'''
	instruction_to_ibm(qc, basis_gates, H0)
	instruction_to_ibm(qc, basis_gates, CX01)
	instruction_to_ibm(qc, basis_gates, CX12)


def algorithm3(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_johannesburg-6,fake_johannesburg-13,fake_johannesburg-14,fake_johannesburg-18,fake_johannesburg-24,fake_johannesburg-31,fake_almaden-16,fake_almaden-19,fake_almaden-20,fake_almaden-22,fake_almaden-23,fake_boeblingen-7,fake_boeblingen-14,fake_boeblingen-15,fake_boeblingen-27,fake_boeblingen-34,fake_poughkeepsie-2,fake_poughkeepsie-3,fake_poughkeepsie-15,fake_poughkeepsie-18,fake_poughkeepsie-22,fake_poughkeepsie-24,fake_poughkeepsie-27,fake_poughkeepsie-30,fake_singapore-5,fake_singapore-10,fake_singapore-22,fake_singapore-23,fake_singapore-32,fake_tokyo-11,fake_tokyo-12,fake_tokyo-13,fake_tokyo-14,fake_tokyo-16,fake_tokyo-17,fake_tokyo-18,fake_tokyo-21,fake_tokyo-23,fake_tokyo-25,fake_tokyo-29,fake_tokyo-30,fake_tokyo-34,fake_tokyo-35,fake_tokyo-45,fake_tokyo-47,fake_tokyo-61,fake_tokyo-62,fake_tokyo-75,fake_hanoi-5,fake_hanoi-8,fake_hanoi-9,fake_hanoi-10,fake_hanoi-15,fake_hanoi-23,fake_hanoi-25,fake_hanoi-28,fake_hanoi-34,fake_hanoi-35,fake_hanoi-36,fake_cairo-12,fake_cairo-18,fake_cairo-21,fake_cairo-26,fake_cairo-27,fake_cairo-28,fake_cairo-32,fake_mumbai-2,fake_mumbai-4,fake_mumbai-11,fake_mumbai-14,fake_mumbai-16,fake_mumbai-22,fake_mumbai-26,fake_mumbai-27,fake_mumbai-34,fake_mumbai-35,fake_mumbai-36,fake_kolkata-2,fake_kolkata-9,fake_kolkata-18,fake_kolkata-25,fake_kolkata-28,fake_kolkata-34,fake_auckland-0,fake_auckland-1,fake_auckland-2,fake_auckland-6,fake_auckland-15,fake_auckland-18,fake_auckland-25,fake_auckland-26,fake_auckland-28,fake_auckland-34,fake_geneva-2,fake_geneva-8,fake_geneva-9,fake_geneva-10,fake_geneva-11,fake_geneva-27,fake_geneva-29,fake_montreal-2,fake_montreal-12,fake_montreal-15,fake_montreal-26,fake_montreal-27,fake_montreal-35,fake_paris-1,fake_paris-6,fake_paris-8,fake_paris-9,fake_paris-10,fake_paris-19,fake_paris-28,fake_paris-31,fake_paris-36,fake_sydney-5,fake_sydney-8,fake_sydney-14,fake_sydney-19,fake_sydney-21,fake_sydney-29,fake_sydney-31,fake_sydney-35,fake_toronto-2,fake_toronto-6,fake_toronto-10,fake_toronto-19,fake_toronto-21,fake_toronto-26,fake_toronto-27,fake_toronto-34,fake_toronto-35,fake_brooklyn-4,fake_brooklyn-12,fake_brooklyn-18,fake_brooklyn-22,fake_brooklyn-23,fake_brooklyn-24,fake_brooklyn-28,fake_brooklyn-29,fake_brooklyn-30,fake_brooklyn-37,fake_brooklyn-43,fake_brooklyn-45,fake_brooklyn-46,fake_brooklyn-49,fake_brooklyn-53,fake_brooklyn-54,fake_brooklyn-57,fake_brooklyn-62,fake_brooklyn-68,fake_brooklyn-78,fake_brooklyn-79,fake_brooklyn-85,fake_brooklyn-88,fake_manhattan-7,fake_manhattan-16,fake_manhattan-20,fake_manhattan-22,fake_manhattan-23,fake_manhattan-31,fake_manhattan-37,fake_manhattan-41,fake_manhattan-43,fake_manhattan-53,fake_manhattan-71,fake_manhattan-74,fake_manhattan-78,fake_manhattan-79,fake_manhattan-89,fake_manhattan-94,fake_cambridge-3,fake_cambridge-5,fake_cambridge-10,fake_cambridge-16,fake_cambridge-17,fake_cambridge-18,fake_cambridge-24,fake_cambridge-27,fake_cambridge-30,fake_cambridge-32,fake_cambridge-33,fake_guadalupe-2,fake_guadalupe-6,fake_guadalupe-15,fake_guadalupe-16,fake_melbourne-0,fake_melbourne-13,fake_melbourne-14,fake_melbourne-18,fake_melbourne-23,fake_melbourne-24,fake_melbourne-27,fake_melbourne-29,fake_melbourne-30,fake_melbourne-32,fake_melbourne-33,fake_rochester-10,fake_rochester-15,fake_rochester-21,fake_rochester-22,fake_rochester-24,fake_rochester-37,fake_rochester-48,fake_rochester-52,fake_rochester-57,fake_rochester-60,fake_rochester-67,fake_rochester-69,fake_rochester-70,fake_washington-0,fake_washington-7,fake_washington-13,fake_washington-15,fake_washington-26,fake_washington-27,fake_washington-34,fake_washington-40,fake_washington-42,fake_washington-43,fake_washington-47,fake_washington-48,fake_washington-50,fake_washington-52,fake_washington-56,fake_washington-58,fake_washington-60,fake_washington-70,fake_washington-71,fake_washington-72,fake_washington-74,fake_washington-75,fake_washington-79,fake_washington-84,fake_washington-89,fake_washington-96,fake_washington-106,fake_washington-108,fake_washington-109,fake_washington-110,fake_washington-112,fake_washington-120,fake_washington-124,fake_washington-132,fake_washington-133,fake_washington-134,fake_washington-143,fake_washington-144,fake_washington-152,fake_washington-157,fake_washington-158,fake_washington-159,fake_washington-167,fake_washington-169,fake_washington-173,fake_washington-180,fake_washington-181,fake_washington-185,fake_washington-190'''
	instruction_to_ibm(qc, basis_gates, H1)
	instruction_to_ibm(qc, basis_gates, CX10)
	instruction_to_ibm(qc, basis_gates, CX12)


def algorithm4(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_johannesburg-7,fake_johannesburg-28,fake_almaden-9,fake_almaden-21,fake_boeblingen-18,fake_singapore-21,fake_tokyo-24,fake_tokyo-65,fake_hanoi-33,fake_cairo-24,fake_mumbai-7,fake_mumbai-17,fake_mumbai-33,fake_kolkata-24,fake_geneva-7,fake_paris-24,fake_paris-33,fake_brooklyn-38,fake_brooklyn-40,fake_brooklyn-63,fake_manhattan-17,fake_manhattan-38,fake_manhattan-63,fake_manhattan-84,fake_manhattan-86,fake_cambridge-26,fake_melbourne-3,fake_rochester-9,fake_rochester-61,fake_washington-23,fake_washington-55,fake_washington-87'''
	instruction_to_ibm(qc, basis_gates, H2)
	instruction_to_ibm(qc, basis_gates, CX21)
	instruction_to_ibm(qc, basis_gates, CX20)


def algorithm5(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_johannesburg-9,fake_johannesburg-15,fake_johannesburg-29,fake_johannesburg-30,fake_almaden-0,fake_almaden-10,fake_boeblingen-2,fake_boeblingen-20,fake_boeblingen-23,fake_boeblingen-28,fake_boeblingen-29,fake_boeblingen-33,fake_poughkeepsie-4,fake_poughkeepsie-11,fake_poughkeepsie-12,fake_poughkeepsie-29,fake_singapore-1,fake_singapore-7,fake_singapore-28,fake_singapore-29,fake_tokyo-2,fake_tokyo-3,fake_tokyo-31,fake_tokyo-32,fake_tokyo-41,fake_tokyo-44,fake_tokyo-50,fake_tokyo-56,fake_tokyo-70,fake_hanoi-2,fake_hanoi-4,fake_hanoi-6,fake_hanoi-12,fake_hanoi-18,fake_hanoi-22,fake_hanoi-32,fake_cairo-8,fake_cairo-11,fake_cairo-16,fake_cairo-23,fake_cairo-29,fake_cairo-34,fake_cairo-36,fake_mumbai-0,fake_mumbai-5,fake_mumbai-12,fake_mumbai-23,fake_mumbai-31,fake_mumbai-32,fake_kolkata-0,fake_kolkata-4,fake_kolkata-14,fake_kolkata-21,fake_kolkata-31,fake_kolkata-32,fake_kolkata-35,fake_auckland-5,fake_auckland-8,fake_auckland-12,fake_auckland-23,fake_auckland-35,fake_geneva-5,fake_geneva-6,fake_geneva-16,fake_geneva-21,fake_geneva-25,fake_geneva-28,fake_geneva-31,fake_geneva-34,fake_montreal-0,fake_montreal-1,fake_montreal-8,fake_montreal-23,fake_montreal-32,fake_paris-4,fake_paris-26,fake_paris-27,fake_paris-29,fake_sydney-2,fake_sydney-4,fake_sydney-11,fake_sydney-15,fake_sydney-25,fake_sydney-26,fake_sydney-27,fake_sydney-32,fake_sydney-34,fake_sydney-36,fake_toronto-0,fake_toronto-4,fake_toronto-5,fake_toronto-14,fake_toronto-23,fake_brooklyn-3,fake_brooklyn-7,fake_brooklyn-9,fake_brooklyn-21,fake_brooklyn-26,fake_brooklyn-27,fake_brooklyn-41,fake_brooklyn-55,fake_brooklyn-64,fake_brooklyn-72,fake_brooklyn-75,fake_brooklyn-80,fake_brooklyn-83,fake_brooklyn-91,fake_manhattan-2,fake_manhattan-9,fake_manhattan-10,fake_manhattan-12,fake_manhattan-21,fake_manhattan-24,fake_manhattan-29,fake_manhattan-35,fake_manhattan-36,fake_manhattan-49,fake_manhattan-52,fake_manhattan-56,fake_manhattan-64,fake_manhattan-70,fake_manhattan-72,fake_manhattan-75,fake_manhattan-77,fake_manhattan-83,fake_manhattan-85,fake_manhattan-90,fake_manhattan-91,fake_cambridge-4,fake_cambridge-6,fake_cambridge-8,fake_cambridge-14,fake_cambridge-15,fake_cambridge-20,fake_cambridge-22,fake_cambridge-25,fake_cambridge-29,fake_guadalupe-0,fake_guadalupe-9,fake_guadalupe-10,fake_guadalupe-12,fake_melbourne-1,fake_melbourne-4,fake_melbourne-5,fake_melbourne-8,fake_melbourne-9,fake_melbourne-10,fake_melbourne-15,fake_melbourne-19,fake_melbourne-20,fake_melbourne-31,fake_melbourne-34,fake_melbourne-35,fake_rochester-8,fake_rochester-18,fake_rochester-20,fake_rochester-23,fake_rochester-30,fake_rochester-45,fake_rochester-50,fake_rochester-56,fake_rochester-58,fake_rochester-71,fake_washington-12,fake_washington-32,fake_washington-33,fake_washington-36,fake_washington-37,fake_washington-41,fake_washington-44,fake_washington-49,fake_washington-63,fake_washington-69,fake_washington-77,fake_washington-86,fake_washington-95,fake_washington-102,fake_washington-104,fake_washington-105,fake_washington-111,fake_washington-114,fake_washington-123,fake_washington-131,fake_washington-136,fake_washington-146,fake_washington-153,fake_washington-156,fake_washington-160,fake_washington-165,fake_washington-166,fake_washington-172,fake_washington-179,fake_washington-182'''
	instruction_to_ibm(qc, basis_gates, H2)
	instruction_to_ibm(qc, basis_gates, CX21)
	instruction_to_ibm(qc, basis_gates, CX10)


def algorithm6(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_johannesburg-16,fake_johannesburg-23,fake_boeblingen-25,fake_boeblingen-30,fake_poughkeepsie-1,fake_poughkeepsie-16,fake_poughkeepsie-23,fake_singapore-3,fake_singapore-8,fake_singapore-30,fake_tokyo-7,fake_tokyo-8,fake_tokyo-68,fake_tokyo-71,fake_tokyo-79,fake_geneva-20,fake_montreal-30,fake_sydney-13,fake_sydney-30,fake_manhattan-1,fake_manhattan-51,fake_melbourne-2,fake_melbourne-6,fake_melbourne-11,fake_melbourne-16,fake_melbourne-21,fake_rochester-47,fake_washington-38,fake_washington-59,fake_washington-73,fake_washington-142,fake_washington-155'''
	instruction_to_ibm(qc, basis_gates, H2)
	instruction_to_ibm(qc, basis_gates, CX20)
	instruction_to_ibm(qc, basis_gates, CX01)


def algorithm7(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_johannesburg-17,fake_johannesburg-20,fake_almaden-4,fake_almaden-26,fake_almaden-31,fake_boeblingen-31,fake_singapore-4,fake_tokyo-48,fake_tokyo-67,fake_tokyo-74,fake_tokyo-76,fake_tokyo-77,fake_tokyo-78,fake_tokyo-80,fake_mumbai-24,fake_auckland-7,fake_geneva-17,fake_geneva-33,fake_montreal-7,fake_montreal-17,fake_sydney-24,fake_toronto-24,fake_brooklyn-19,fake_brooklyn-82,fake_brooklyn-86,fake_manhattan-82,fake_cambridge-7,fake_cambridge-28,fake_guadalupe-17,fake_rochester-7,fake_rochester-26,fake_rochester-28,fake_washington-57,fake_washington-85,fake_washington-115,fake_washington-119,fake_washington-145,fake_washington-149,fake_washington-176'''
	instruction_to_ibm(qc, basis_gates, H1)
	instruction_to_ibm(qc, basis_gates, CX12)
	instruction_to_ibm(qc, basis_gates, CX20)


def algorithm8(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_almaden-17,fake_boeblingen-17,fake_tokyo-46,fake_tokyo-60,fake_mumbai-3,fake_mumbai-13,fake_kolkata-3,fake_kolkata-30,fake_montreal-3,fake_paris-20,fake_paris-30,fake_sydney-20,fake_toronto-3,fake_toronto-20,fake_brooklyn-25,fake_brooklyn-44,fake_brooklyn-69,fake_manhattan-8,fake_manhattan-25,fake_manhattan-44,fake_cambridge-12,fake_cambridge-19,fake_guadalupe-3,fake_guadalupe-13,fake_melbourne-25,fake_rochester-33,fake_washington-1,fake_washington-8,fake_washington-80,fake_washington-107,fake_washington-121,fake_washington-135,fake_washington-168'''
	instruction_to_ibm(qc, basis_gates, H1)
	instruction_to_ibm(qc, basis_gates, CX10)
	instruction_to_ibm(qc, basis_gates, CX02)


def algorithm9(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_almaden-18,fake_boeblingen-21,fake_boeblingen-26,fake_poughkeepsie-17,fake_poughkeepsie-28,fake_singapore-9,fake_singapore-26,fake_singapore-31,fake_tokyo-9,fake_tokyo-15,fake_tokyo-19,fake_tokyo-20,fake_hanoi-7,fake_hanoi-24,fake_cairo-17,fake_kolkata-7,fake_kolkata-17,fake_kolkata-33,fake_auckland-17,fake_auckland-24,fake_auckland-33,fake_geneva-24,fake_paris-7,fake_sydney-7,fake_sydney-17,fake_toronto-7,fake_toronto-17,fake_toronto-33,fake_brooklyn-17,fake_brooklyn-61,fake_brooklyn-84,fake_manhattan-19,fake_manhattan-40,fake_manhattan-42,fake_cambridge-9,fake_melbourne-7,fake_melbourne-12,fake_melbourne-17,fake_melbourne-22,fake_melbourne-26,fake_melbourne-28,fake_rochester-63,fake_washington-25,fake_washington-51,fake_washington-83,fake_washington-113,fake_washington-147,fake_washington-174,fake_washington-178'''
	instruction_to_ibm(qc, basis_gates, H2)
	instruction_to_ibm(qc, basis_gates, CX20)
	instruction_to_ibm(qc, basis_gates, CX21)


def algorithm10(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_boeblingen-4,fake_boeblingen-9,fake_poughkeepsie-7,fake_poughkeepsie-20,fake_singapore-18,fake_tokyo-53,fake_tokyo-54,fake_hanoi-17,fake_cairo-7,fake_cairo-33,fake_montreal-24,fake_montreal-33,fake_paris-17,fake_sydney-33,fake_brooklyn-42,fake_manhattan-61,fake_guadalupe-7,fake_rochester-42,fake_rochester-44,fake_washington-21,fake_washington-53,fake_washington-117'''
	instruction_to_ibm(qc, basis_gates, H0)
	instruction_to_ibm(qc, basis_gates, CX02)
	instruction_to_ibm(qc, basis_gates, CX21)


def algorithm11(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_boeblingen-8,fake_boeblingen-12,fake_singapore-12,fake_singapore-17,fake_singapore-25,fake_tokyo-37,fake_tokyo-38,fake_tokyo-40,fake_tokyo-66,fake_hanoi-13,fake_cairo-20,fake_cairo-30,fake_mumbai-20,fake_mumbai-30,fake_kolkata-20,fake_geneva-13,fake_geneva-30,fake_montreal-13,fake_paris-3,fake_paris-13,fake_toronto-13,fake_brooklyn-1,fake_brooklyn-51,fake_brooklyn-76,fake_manhattan-58,fake_manhattan-69,fake_manhattan-76,fake_cambridge-1,fake_rochester-1,fake_rochester-19,fake_rochester-54,fake_rochester-68,fake_washington-31'''
	instruction_to_ibm(qc, basis_gates, H0)
	instruction_to_ibm(qc, basis_gates, CX01)
	instruction_to_ibm(qc, basis_gates, CX02)


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
