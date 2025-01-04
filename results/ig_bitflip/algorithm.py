import os, sys, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qiskit import QuantumCircuit, ClassicalRegister
from simulator import QSimulator
from ibm_noise_models import NoiseModel, Instruction, instruction_to_ibm, Op

###### ACTIONS ######
X0 = [Instruction(0, Op.X, None, None)]
CX = [Instruction(2, Op.CNOT, 0, None), Instruction(2, Op.CNOT, 1, None)]
P2 = [Instruction(2, Op.MEAS, None, None)]
def my_algorithm(qc: QuantumCircuit, initial_state, noise_model: NoiseModel, seed=1):
	#### INITIALIZE SIMULATOR ######
	qs, cs = initial_state
	simulator = QSimulator(noise_model, seed)
	simulator.qmemory = qs
	simulator.meas_cache = cs
	current_state = 14

	while True:
		if current_state in {0, 1, 2, 3}:
			# target state reached
			break
		if current_state == 14:
			actions = ['CX']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX and simulator.meas_cache.get_memory_val() == 0:
				current_state = 10
				continue
			raise Exception('Invalid (classical) memory state at 14')
		if current_state == 10:
			actions = ['P2']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == P2 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 3
				continue
			if choosen_action == P2 and simulator.meas_cache.get_memory_val() == 4:
				current_state = 6
				continue
			raise Exception('Invalid (classical) memory state at 10')
		if current_state == 6:
			actions = ['X0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == X0 and simulator.meas_cache.get_memory_val() == 4:
				current_state = 1
				continue
			raise Exception('Invalid (classical) memory state at 6')
