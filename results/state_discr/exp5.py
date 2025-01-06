import os, sys, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qiskit import QuantumCircuit, ClassicalRegister
from simulator import QSimulator
from ibm_noise_models import NoiseModel, Instruction, instruction_to_ibm, Op

###### ACTIONS ######
H0 = [Instruction(0, Op.H, None, None)]
S0 = [Instruction(0, Op.S, None, None)]
CX01 = [Instruction(1, Op.CNOT, 0, None)]
H1 = [Instruction(1, Op.H, None, None)]
S1 = [Instruction(1, Op.S, None, None)]
CX10 = [Instruction(0, Op.CNOT, 1, None)]
MEAS1 = [Instruction(1, Op.MEAS, None, None)]
IS0 = [Instruction(2, Op.WRITE1, None, None)]
ISPlus = [Instruction(3, Op.WRITE1, None, None)]
CAncilla0 = [Instruction(4, Op.WRITE0, None, None)]
CAncilla1 = [Instruction(4, Op.WRITE1, None, None)]
CAncilla20 = [Instruction(5, Op.WRITE0, None, None)]
CAncilla21 = [Instruction(5, Op.WRITE1, None, None)]
def my_algorithm(qc: QuantumCircuit, initial_state, noise_model: NoiseModel, seed=1):
	#### INITIALIZE SIMULATOR ######
	qs, cs = initial_state
	simulator = QSimulator(noise_model, seed)
	simulator.qmemory = qs
	simulator.meas_cache = cs
	current_state = None

	while True:
		if current_state in set():
			# target state reached
			break
