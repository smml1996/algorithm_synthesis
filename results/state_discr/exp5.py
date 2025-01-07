import os, sys, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qiskit import QuantumCircuit, ClassicalRegister
from simulator import QSimulator
from ibm_noise_models import NoiseModel, Instruction, instruction_to_ibm, Op

###### ACTIONS ######
H0 = [Instruction(0, Op.H, None, None)]
MEAS0 = [Instruction(0, Op.MEAS, None, None)]
RESET = [Instruction(0, Op.RESET, None, None), Instruction(0, Op.CH, 1, None), Instruction(0, Op.CNOT, 2, None)]
IS0 = [Instruction(1, Op.WRITE1, None, None)]
ISPlus = [Instruction(2, Op.WRITE1, None, None)]
IS1 = [Instruction(1, Op.WRITE1, None, None), Instruction(2, Op.WRITE1, None, None)]
def my_algorithm(qc: QuantumCircuit, initial_state, noise_model: NoiseModel, seed=1):
	#### INITIALIZE SIMULATOR ######
	qs, cs = initial_state
	simulator = QSimulator(noise_model, seed)
	simulator.qmemory = qs
	simulator.meas_cache = cs
	current_state = 93

	while True:
		if current_state in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}:
			# target state reached
			break
		if current_state == 93:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 81
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 75
				continue
			raise Exception('Invalid (classical) memory state at 93')
		if current_state == 81:
			actions = ['RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 0:
				current_state = 56
				continue
			raise Exception('Invalid (classical) memory state at 81')
		if current_state == 75:
			actions = ['RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 57
				continue
			raise Exception('Invalid (classical) memory state at 75')
		if current_state == 56:
			actions = ['H0', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 46
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 81
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 30
				continue
			raise Exception('Invalid (classical) memory state at 56')
		if current_state == 57:
			actions = ['H0', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 55
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 29
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 75
				continue
			raise Exception('Invalid (classical) memory state at 57')
		if current_state == 46:
			actions = ['H0', 'MEAS0', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 56
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 81
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 36
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 0:
				current_state = 56
				continue
			raise Exception('Invalid (classical) memory state at 46')
		if current_state == 55:
			actions = ['H0', 'MEAS0', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 57
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 73
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 23
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 57
				continue
			raise Exception('Invalid (classical) memory state at 55')
		if current_state == 29:
			actions = ['ISPlus']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == ISPlus and simulator.meas_cache.get_memory_val() == 4:
				current_state = 14
				continue
			raise Exception('Invalid (classical) memory state at 29')
		if current_state == 30:
			actions = ['ISPlus']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == ISPlus and simulator.meas_cache.get_memory_val() == 5:
				current_state = 17
				continue
			raise Exception('Invalid (classical) memory state at 30')
		if current_state == 36:
			actions = ['IS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == IS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 15
				continue
			raise Exception('Invalid (classical) memory state at 36')
		if current_state == 73:
			actions = ['RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 0:
				current_state = 50
				continue
			raise Exception('Invalid (classical) memory state at 73')
		if current_state == 23:
			actions = ['IS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == IS1 and simulator.meas_cache.get_memory_val() == 7:
				current_state = 2
				continue
			raise Exception('Invalid (classical) memory state at 23')
		if current_state == 50:
			actions = ['H0', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 61
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 29
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 75
				continue
			raise Exception('Invalid (classical) memory state at 50')
		if current_state == 61:
			actions = ['H0', 'MEAS0', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 50
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 73
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 23
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 0:
				current_state = 50
				continue
			raise Exception('Invalid (classical) memory state at 61')
