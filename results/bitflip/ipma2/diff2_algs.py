import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qiskit import QuantumCircuit, ClassicalRegister
from ibm_noise_models import Instruction, instruction_to_ibm

def algorithm0(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_johannesburg-0,fake_almaden-0,fake_boeblingen-1,fake_boeblingen-2,fake_poughkeepsie-1,fake_poughkeepsie-2,fake_singapore-1,fake_tokyo-3,fake_hanoi-0,fake_hanoi-1,fake_hanoi-3,fake_cairo-1,fake_mumbai-2,fake_mumbai-3,fake_kolkata-2,fake_kolkata-3,fake_montreal-0,fake_paris-1,fake_cambridge-1,fake_cambridge-2,fake_cambridge-3,fake_melbourne-0,fake_rochester-1,fake_rochester-3,fake_washington-0,fake_washington-1,fake_washington-2'''
	CX
	CX


def algorithm1(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_johannesburg-1,fake_johannesburg-2,fake_boeblingen-0,fake_boeblingen-3,fake_poughkeepsie-3,fake_singapore-3,fake_tokyo-2,fake_cairo-0,fake_cairo-3,fake_mumbai-0,fake_kolkata-0,fake_geneva-1,fake_montreal-2,fake_toronto-0,fake_brooklyn-2,fake_brooklyn-3,fake_melbourne-1,fake_melbourne-2'''
	X0
	X0


def algorithm2(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_johannesburg-3,fake_almaden-1,fake_almaden-2,fake_almaden-3,fake_poughkeepsie-0,fake_singapore-0,fake_singapore-2,fake_tokyo-0,fake_tokyo-1,fake_tokyo-4,fake_hanoi-2,fake_cairo-2,fake_mumbai-1,fake_kolkata-1,fake_auckland-0,fake_auckland-1,fake_geneva-0,fake_montreal-1,fake_paris-0,fake_paris-2,fake_sydney-0,fake_sydney-1,fake_toronto-1,fake_toronto-2,fake_brooklyn-0,fake_brooklyn-1,fake_manhattan-0,fake_manhattan-1,fake_manhattan-2,fake_manhattan-3,fake_manhattan-4,fake_cambridge-0,fake_guadalupe-0,fake_guadalupe-1,fake_melbourne-3,fake_rochester-0,fake_rochester-2,fake_rochester-4'''
	halt


algorithms = []
algorithms.append(algorithm0)
algorithms.append(algorithm1)
algorithms.append(algorithm2)
