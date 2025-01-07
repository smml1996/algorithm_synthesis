import os, sys, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qiskit import QuantumCircuit, ClassicalRegister
from simulator import QSimulator
from ibm_noise_models import NoiseModel, Instruction, instruction_to_ibm, Op

###### ACTIONS ######
H0 = [Instruction(0, Op.H, None, None)]
MEAS0 = [Instruction(0, Op.MEAS, None, None)]
RESET = [Instruction(0, Op.RESET, None, None), Instruction(0, Op.CH, 1, None), Instruction(0, Op.CNOT, 2, None), Instruction(0, Op.CNOT, 3, None), Instruction(0, Op.CH, 3, None)]
IS0 = [Instruction(1, Op.WRITE1, None, None)]
ISPlus = [Instruction(2, Op.WRITE1, None, None)]
IS1 = [Instruction(1, Op.WRITE1, None, None), Instruction(2, Op.WRITE1, None, None)]
ISMinus = [Instruction(3, Op.WRITE1, None, None)]
def my_algorithm(qc: QuantumCircuit, initial_state, noise_model: NoiseModel, seed=1):
	#### INITIALIZE SIMULATOR ######
	qs, cs = initial_state
	simulator = QSimulator(noise_model, seed)
	simulator.qmemory = qs
	simulator.meas_cache = cs
	current_state = 101

	while True:
		if current_state in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}:
			# target state reached
			break
		if current_state == 101:
			actions = ['H0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 90
				continue
			raise Exception('Invalid (classical) memory state at 101')
		if current_state == 90:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 87
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 72
				continue
			raise Exception('Invalid (classical) memory state at 90')
		if current_state == 87:
			actions = ['RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 0:
				current_state = 53
				continue
			raise Exception('Invalid (classical) memory state at 87')
		if current_state == 72:
			actions = ['RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 52
				continue
			raise Exception('Invalid (classical) memory state at 72')
		if current_state == 53:
			actions = ['MEAS0', 'H0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 87
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 19
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 51
				continue
			raise Exception('Invalid (classical) memory state at 53')
		if current_state == 52:
			actions = ['MEAS0', 'H0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 83
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 21
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 63
				continue
			raise Exception('Invalid (classical) memory state at 52')
		if current_state == 51:
			actions = ['MEAS0', 'RESET', 'H0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 87
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 30
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 0:
				current_state = 53
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 53
				continue
			raise Exception('Invalid (classical) memory state at 51')
		if current_state == 83:
			actions = ['RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 0:
				current_state = 48
				continue
			raise Exception('Invalid (classical) memory state at 83')
		if current_state == 63:
			actions = ['MEAS0', 'RESET', 'H0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 41
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 72
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 52
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 52
				continue
			raise Exception('Invalid (classical) memory state at 63')
		if current_state == 19:
			actions = ['ISPlus']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == ISPlus and simulator.meas_cache.get_memory_val() == 5:
				current_state = 4
				continue
			raise Exception('Invalid (classical) memory state at 19')
		if current_state == 21:
			actions = ['ISMinus']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == ISMinus and simulator.meas_cache.get_memory_val() == 9:
				current_state = 3
				continue
			raise Exception('Invalid (classical) memory state at 21')
		if current_state == 30:
			actions = ['IS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == IS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 11
				continue
			raise Exception('Invalid (classical) memory state at 30')
		if current_state == 48:
			actions = ['MEAS0', 'H0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 83
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 21
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 44
				continue
			raise Exception('Invalid (classical) memory state at 48')
		if current_state == 41:
			actions = ['IS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == IS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 16
				continue
			raise Exception('Invalid (classical) memory state at 41')
		if current_state == 44:
			actions = ['MEAS0', 'RESET', 'H0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 41
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 72
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 0:
				current_state = 48
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 48
				continue
			raise Exception('Invalid (classical) memory state at 44')
