import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qiskit import QuantumCircuit, ClassicalRegister
from utils import Instruction, instruction_to_ibm

def algorithm0(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_athens-0,fake_athens-1,fake_athens-2,fake_belem-0,fake_belem-1,fake_belem-2,fake_tenerife-0,fake_tenerife-1,fake_lima-0,fake_lima-2,fake_rome-0,fake_manila-0,fake_manila-1,fake_manila-2,fake_santiago-2,fake_bogota-0,fake_ourense-0,fake_ourense-1,fake_ourense-2,fake_essex-0,fake_essex-1,fake_essex-2,fake_vigo-0,fake_vigo-1,fake_vigo-2,fake_burlington-1,fake_burlington-2,fake_quito-2,fake_jakarta-0,fake_jakarta-1,fake_jakarta-2,fake_jakarta-3,fake_oslo-1,fake_oslo-2,fake_oslo-4,fake_perth-0,fake_perth-1,fake_perth-2,fake_lagos-0,fake_lagos-1,fake_lagos-2,fake_lagos-3,fake_lagos-4,fake_nairobi-1,fake_nairobi-3,fake_nairobi-4,fake_casablanca-1,fake_casablanca-2,fake_casablanca-3,fake_casablanca-4,fake_melbourne-1,fake_melbourne-2,fake_melbourne-3,fake_melbourne-4,fake_melbourne-5,fake_melbourne-6,fake_melbourne-7,fake_guadalupe-0,fake_guadalupe-1,fake_guadalupe-3,fake_tokyo-0,fake_tokyo-1,fake_tokyo-2,fake_tokyo-5,fake_tokyo-6,fake_tokyo-7,fake_tokyo-8,fake_tokyo-9,fake_poughkeepsie-0,fake_poughkeepsie-3,fake_poughkeepsie-4,fake_johannesburg-0,fake_johannesburg-1,fake_johannesburg-2,fake_boeblingen-0,fake_boeblingen-3,fake_boeblingen-5,fake_almaden-0,fake_almaden-1,fake_almaden-2,fake_almaden-4,fake_almaden-5,fake_almaden-6,fake_singapore-0,fake_singapore-1,fake_singapore-2,fake_singapore-3,fake_singapore-4,fake_singapore-5,fake_mumbai-0,fake_mumbai-1,fake_mumbai-5,fake_paris-0,fake_paris-4,fake_paris-5,fake_paris-6,fake_auckland-0,fake_auckland-1,fake_kolkata-0,fake_kolkata-3,fake_kolkata-4,fake_kolkata-5,fake_toronto-0,fake_toronto-1,fake_toronto-2,fake_toronto-3,fake_toronto-4,fake_toronto-8,fake_montreal-1,fake_montreal-2,fake_montreal-3,fake_sydney-0,fake_sydney-1,fake_sydney-3,fake_sydney-5,fake_cairo-0,fake_cairo-1,fake_cairo-2,fake_cairo-3,fake_cairo-7,fake_cairo-8,fake_hanoi-0,fake_hanoi-1,fake_hanoi-3,fake_hanoi-4,fake_hanoi-6,fake_geneva-0,fake_geneva-2,fake_cambridge-0,fake_cambridge-2,fake_rochester-4,fake_brooklyn-0,fake_brooklyn-2,fake_brooklyn-3,fake_brooklyn-4,fake_manhattan-0,fake_manhattan-1,fake_manhattan-3,fake_manhattan-4,fake_manhattan-5,fake_washington-4,fake_washington-5,fake_washington-6'''
	instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 1, 2)
	instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 1, 2)
	instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 1, 2)
	instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 1, 2)


def algorithm1(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_yorktown-2,fake_yorktown-3,fake_quito-0,fake_quito-1,fake_london-2,fake_oslo-0,fake_perth-3,fake_perth-4,fake_tokyo-3,fake_poughkeepsie-1,fake_poughkeepsie-2,fake_boeblingen-2,fake_boeblingen-4,fake_boeblingen-6,fake_mumbai-4,fake_mumbai-6,fake_kolkata-2,fake_montreal-0,fake_cairo-9,fake_cambridge-1,fake_cambridge-3,fake_cambridge-4,fake_rochester-0,fake_manhattan-2,fake_manhattan-6,fake_washington-2,fake_washington-3'''
	instruction_to_ibm(qc, basis_gates, Instruction.H, 1, None)
	instruction_to_ibm(qc, basis_gates, Instruction.H, 1, None)
	instruction_to_ibm(qc, basis_gates, Instruction.H, 1, None)
	instruction_to_ibm(qc, basis_gates, Instruction.H, 1, None)


algorithms = []
algorithms.append(algorithm0)
algorithms.append(algorithm1)
