import os, sys, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qiskit import QuantumCircuit, ClassicalRegister
from simulator import QSimulator
from ibm_noise_models import NoiseModel, Instruction, instruction_to_ibm, Op

###### ACTIONS ######
H0 = [Instruction(0, Op.H, None, None)]
MEAS0 = [Instruction(0, Op.MEAS, None, None)]
RESET = [Instruction(0, Op.RESET, None, None), Instruction(0, Op.CH, 1, None)]
IS0 = [Instruction(1, Op.WRITE1, None, None)]
ISPlus = [Instruction(2, Op.WRITE1, None, None)]
def my_algorithm(qc: QuantumCircuit, initial_state, noise_model: NoiseModel, seed=1):
	#### INITIALIZE SIMULATOR ######
	qs, cs = initial_state
	simulator = QSimulator(noise_model, seed)
	simulator.qmemory = qs
	simulator.meas_cache = cs
	current_state = 27

	while True:
		if current_state in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}:
			# target state reached
			break
		if current_state == 27:
			actions = ['RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 0:
				current_state = 27
				continue
			raise Exception('Invalid (classical) memory state at 27')
