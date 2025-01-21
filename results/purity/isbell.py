import os, sys, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qiskit import QuantumCircuit, ClassicalRegister
from simulator import QSimulator
from ibm_noise_models import NoiseModel, Instruction, instruction_to_ibm, Op

###### ACTIONS ######
MEAS0 = [Instruction(0, Op.MEAS, None, None)]
H0 = [Instruction(0, Op.H, None, None)]
CX01 = [Instruction(1, Op.CNOT, 0, None)]
MEAS1 = [Instruction(1, Op.MEAS, None, None)]
H1 = [Instruction(1, Op.H, None, None)]
CX10 = [Instruction(0, Op.CNOT, 1, None)]
RESET = [Instruction(0, Op.RESET, None, None), Instruction(1, Op.RESET, None, None), Instruction(0, Op.CH, 2, None), Instruction(1, Op.CNOT, 0, None)]
NOTEntangled = [Instruction(2, Op.WRITE0, None, None), Instruction(3, Op.WRITE1, None, None)]
Entangled = [Instruction(2, Op.WRITE1, None, None), Instruction(3, Op.WRITE0, None, None)]
def my_algorithm(qc: QuantumCircuit, initial_state, noise_model: NoiseModel, seed=1):
	#### INITIALIZE SIMULATOR ######
	qs, cs = initial_state
	simulator = QSimulator(noise_model, seed)
	simulator.qmemory = qs
	simulator.meas_cache = cs
	current_state = 569

	while True:
		if current_state in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191}:
			# target state reached
			break
		if current_state == 569:
			actions = ['MEAS0', 'CX01', 'CX10', 'MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 893
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 329
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 595
				continue
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 515
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 893
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 225
				continue
			raise Exception('Invalid (classical) memory state at 569')
		if current_state == 893:
			actions = ['RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 0:
				current_state = 569
				continue
			raise Exception('Invalid (classical) memory state at 893')
		if current_state == 595:
			actions = ['H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 678
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 569
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 497
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 0:
				current_state = 569
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 893
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 320
				continue
			raise Exception('Invalid (classical) memory state at 595')
		if current_state == 515:
			actions = ['CX10', 'MEAS1', 'H0', 'H1', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 569
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 893
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 275
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 712
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 612
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 0:
				current_state = 569
				continue
			raise Exception('Invalid (classical) memory state at 515')
		if current_state == 329:
			actions = ['Entangled']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == Entangled and simulator.meas_cache.get_memory_val() == 5:
				current_state = 116
				continue
			raise Exception('Invalid (classical) memory state at 329')
		if current_state == 225:
			actions = ['Entangled']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == Entangled and simulator.meas_cache.get_memory_val() == 6:
				current_state = 129
				continue
			raise Exception('Invalid (classical) memory state at 225')
		if current_state == 678:
			actions = ['H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 595
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 676
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 703
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 0:
				current_state = 569
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 291
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 893
				continue
			raise Exception('Invalid (classical) memory state at 678')
		if current_state == 497:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 462
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 595
				continue
			raise Exception('Invalid (classical) memory state at 497')
		if current_state == 712:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 559
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 515
				continue
			raise Exception('Invalid (classical) memory state at 712')
		if current_state == 612:
			actions = ['CX10', 'MEAS1', 'H0', 'H1', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 676
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 285
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 893
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 730
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 515
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 0:
				current_state = 569
				continue
			raise Exception('Invalid (classical) memory state at 612')
		if current_state == 320:
			actions = ['Entangled']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == Entangled and simulator.meas_cache.get_memory_val() == 5:
				current_state = 69
				continue
			raise Exception('Invalid (classical) memory state at 320')
		if current_state == 275:
			actions = ['Entangled']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == Entangled and simulator.meas_cache.get_memory_val() == 6:
				current_state = 23
				continue
			raise Exception('Invalid (classical) memory state at 275')
		if current_state == 676:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 612
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 240
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 893
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 600
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 678
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 508
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 0:
				current_state = 569
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 893
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 197
				continue
			raise Exception('Invalid (classical) memory state at 676')
		if current_state == 703:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 523
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 678
				continue
			raise Exception('Invalid (classical) memory state at 703')
		if current_state == 291:
			actions = ['NOTEntangled']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == NOTEntangled and simulator.meas_cache.get_memory_val() == 9:
				current_state = 78
				continue
			raise Exception('Invalid (classical) memory state at 291')
		if current_state == 462:
			actions = ['CX10', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 817
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 523
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 418
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 590
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 962
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 374
				continue
			raise Exception('Invalid (classical) memory state at 462')
		if current_state == 559:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 638
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 1027
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 374
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 812
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 619
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 551
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			raise Exception('Invalid (classical) memory state at 559')
		if current_state == 285:
			actions = ['NOTEntangled']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == NOTEntangled and simulator.meas_cache.get_memory_val() == 10:
				current_state = 117
				continue
			raise Exception('Invalid (classical) memory state at 285')
		if current_state == 730:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 551
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 612
				continue
			raise Exception('Invalid (classical) memory state at 730')
		if current_state == 240:
			actions = ['NOTEntangled']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == NOTEntangled and simulator.meas_cache.get_memory_val() == 10:
				current_state = 26
				continue
			raise Exception('Invalid (classical) memory state at 240')
		if current_state == 600:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 747
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 612
				continue
			raise Exception('Invalid (classical) memory state at 600')
		if current_state == 508:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 602
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 678
				continue
			raise Exception('Invalid (classical) memory state at 508')
		if current_state == 197:
			actions = ['NOTEntangled']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == NOTEntangled and simulator.meas_cache.get_memory_val() == 9:
				current_state = 190
				continue
			raise Exception('Invalid (classical) memory state at 197')
		if current_state == 523:
			actions = ['CX10', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 514
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 462
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 816
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 404
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 204
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 962
				continue
			raise Exception('Invalid (classical) memory state at 523')
		if current_state == 817:
			actions = ['CX10', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 462
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 602
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 650
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 485
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 991
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 275
				continue
			raise Exception('Invalid (classical) memory state at 817')
		if current_state == 418:
			actions = ['MEAS0', 'CX01', 'CX10', 'MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 962
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 302
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 462
				continue
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 686
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 962
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 221
				continue
			raise Exception('Invalid (classical) memory state at 418')
		if current_state == 590:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 462
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 595
				continue
			raise Exception('Invalid (classical) memory state at 590')
		if current_state == 828:
			actions = ['MEAS0', 'CX01', 'CX10', 'MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 841
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 374
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 456
				continue
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 808
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 893
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 225
				continue
			raise Exception('Invalid (classical) memory state at 828')
		if current_state == 962:
			actions = ['RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			raise Exception('Invalid (classical) memory state at 962')
		if current_state == 638:
			actions = ['MEAS0', 'CX01', 'CX10', 'MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 1027
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 352
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 831
				continue
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 559
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 1027
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 195
				continue
			raise Exception('Invalid (classical) memory state at 638')
		if current_state == 1027:
			actions = ['RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			raise Exception('Invalid (classical) memory state at 1027')
		if current_state == 812:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 559
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 515
				continue
			raise Exception('Invalid (classical) memory state at 812')
		if current_state == 619:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 728
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 991
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 320
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 621
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 559
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 747
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			raise Exception('Invalid (classical) memory state at 619')
		if current_state == 551:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 642
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 1027
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 204
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 560
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 635
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 559
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			raise Exception('Invalid (classical) memory state at 551')
		if current_state == 807:
			actions = ['MEAS0', 'CX01', 'CX10', 'MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 893
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 329
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 531
				continue
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 480
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 891
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 374
				continue
			raise Exception('Invalid (classical) memory state at 807')
		if current_state == 374:
			actions = ['Entangled']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == Entangled and simulator.meas_cache.get_memory_val() == 7:
				current_state = 175
				continue
			raise Exception('Invalid (classical) memory state at 374')
		if current_state == 747:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 748
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 1027
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 204
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 461
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 499
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 619
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			raise Exception('Invalid (classical) memory state at 747')
		if current_state == 602:
			actions = ['CX10', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 557
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 817
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 402
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 820
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 204
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 962
				continue
			raise Exception('Invalid (classical) memory state at 602')
		if current_state == 514:
			actions = ['CX10', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 523
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 448
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 463
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 827
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 991
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 285
				continue
			raise Exception('Invalid (classical) memory state at 514')
		if current_state == 816:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 425
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 332
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 962
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 783
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 523
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 477
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 365
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 962
				continue
			raise Exception('Invalid (classical) memory state at 816')
		if current_state == 404:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 523
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 678
				continue
			raise Exception('Invalid (classical) memory state at 404')
		if current_state == 204:
			actions = ['NOTEntangled']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == NOTEntangled and simulator.meas_cache.get_memory_val() == 11:
				current_state = 53
				continue
			raise Exception('Invalid (classical) memory state at 204')
		if current_state == 650:
			actions = ['MEAS0', 'CX01', 'CX10', 'MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 1040
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 275
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 817
				continue
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 814
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 916
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 275
				continue
			raise Exception('Invalid (classical) memory state at 650')
		if current_state == 485:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 614
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 817
				continue
			raise Exception('Invalid (classical) memory state at 485')
		if current_state == 991:
			actions = ['RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 3:
				current_state = 654
				continue
			raise Exception('Invalid (classical) memory state at 991')
		if current_state == 686:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 418
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 1035
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 221
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 437
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 814
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 668
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 643
				continue
			raise Exception('Invalid (classical) memory state at 686')
		if current_state == 841:
			actions = ['RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			raise Exception('Invalid (classical) memory state at 841')
		if current_state == 456:
			actions = ['MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 595
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 618
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 637
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 841
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 302
				continue
			raise Exception('Invalid (classical) memory state at 456')
		if current_state == 808:
			actions = ['CX10', 'MEAS1', 'H0', 'H1', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 893
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 275
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 543
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 616
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			raise Exception('Invalid (classical) memory state at 808')
		if current_state == 831:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 390
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 605
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 673
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 638
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 582
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 968
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 352
				continue
			raise Exception('Invalid (classical) memory state at 831')
		if current_state == 728:
			actions = ['MEAS0', 'CX01', 'CX10', 'MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 1039
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 320
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 390
				continue
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 619
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 950
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 320
				continue
			raise Exception('Invalid (classical) memory state at 728')
		if current_state == 621:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 490
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 619
				continue
			raise Exception('Invalid (classical) memory state at 621')
		if current_state == 642:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 551
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 1027
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 317
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 473
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 420
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 512
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 1027
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 208
				continue
			raise Exception('Invalid (classical) memory state at 642')
		if current_state == 560:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 551
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 612
				continue
			raise Exception('Invalid (classical) memory state at 560')
		if current_state == 635:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 682
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 291
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 991
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 451
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 551
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 588
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			raise Exception('Invalid (classical) memory state at 635')
		if current_state == 531:
			actions = ['H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 818
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 717
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 893
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 320
				continue
			raise Exception('Invalid (classical) memory state at 531')
		if current_state == 480:
			actions = ['CX10', 'MEAS1', 'H0', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 891
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 195
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 695
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 773
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 515
				continue
			raise Exception('Invalid (classical) memory state at 480')
		if current_state == 891:
			actions = ['RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			raise Exception('Invalid (classical) memory state at 891')
		if current_state == 302:
			actions = ['Entangled']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == Entangled and simulator.meas_cache.get_memory_val() == 7:
				current_state = 50
				continue
			raise Exception('Invalid (classical) memory state at 302')
		if current_state == 221:
			actions = ['Entangled']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == Entangled and simulator.meas_cache.get_memory_val() == 4:
				current_state = 5
				continue
			raise Exception('Invalid (classical) memory state at 221')
		if current_state == 352:
			actions = ['Entangled']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == Entangled and simulator.meas_cache.get_memory_val() == 4:
				current_state = 158
				continue
			raise Exception('Invalid (classical) memory state at 352')
		if current_state == 195:
			actions = ['Entangled']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == Entangled and simulator.meas_cache.get_memory_val() == 7:
				current_state = 127
				continue
			raise Exception('Invalid (classical) memory state at 195')
		if current_state == 748:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 747
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 1027
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 317
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 570
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 443
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 488
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 1027
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 208
				continue
			raise Exception('Invalid (classical) memory state at 748')
		if current_state == 461:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 747
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 502
				continue
			raise Exception('Invalid (classical) memory state at 461')
		if current_state == 499:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 566
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 291
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 991
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 762
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 747
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 633
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			raise Exception('Invalid (classical) memory state at 499')
		if current_state == 557:
			actions = ['CX10', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 602
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 613
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 697
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 797
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 991
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 285
				continue
			raise Exception('Invalid (classical) memory state at 557')
		if current_state == 402:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 546
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 332
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 962
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 450
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 602
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 496
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 365
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 962
				continue
			raise Exception('Invalid (classical) memory state at 402')
		if current_state == 820:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 602
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 634
				continue
			raise Exception('Invalid (classical) memory state at 820')
		if current_state == 448:
			actions = ['CX10', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 613
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 514
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 779
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 522
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 962
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 374
				continue
			raise Exception('Invalid (classical) memory state at 448')
		if current_state == 463:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 594
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 285
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 916
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 802
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 514
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 709
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 1040
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 285
				continue
			raise Exception('Invalid (classical) memory state at 463')
		if current_state == 827:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 469
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 514
				continue
			raise Exception('Invalid (classical) memory state at 827')
		if current_state == 425:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 816
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 1035
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 332
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 693
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 594
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 795
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 563
				continue
			raise Exception('Invalid (classical) memory state at 425')
		if current_state == 332:
			actions = ['NOTEntangled']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == NOTEntangled and simulator.meas_cache.get_memory_val() == 8:
				current_state = 83
				continue
			raise Exception('Invalid (classical) memory state at 332')
		if current_state == 783:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 732
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 532
				continue
			raise Exception('Invalid (classical) memory state at 783')
		if current_state == 477:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 602
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 678
				continue
			raise Exception('Invalid (classical) memory state at 477')
		if current_state == 365:
			actions = ['NOTEntangled']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == NOTEntangled and simulator.meas_cache.get_memory_val() == 11:
				current_state = 166
				continue
			raise Exception('Invalid (classical) memory state at 365')
		if current_state == 1040:
			actions = ['RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 3:
				current_state = 654
				continue
			raise Exception('Invalid (classical) memory state at 1040')
		if current_state == 814:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 650
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 916
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 225
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 792
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 686
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 594
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 729
				continue
			raise Exception('Invalid (classical) memory state at 814')
		if current_state == 916:
			actions = ['RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 0:
				current_state = 569
				continue
			raise Exception('Invalid (classical) memory state at 916')
		if current_state == 614:
			actions = ['H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 634
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 657
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 460
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 0:
				current_state = 569
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 1027
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 231
				continue
			raise Exception('Invalid (classical) memory state at 614')
		if current_state == 654:
			actions = ['MEAS0', 'CX01', 'CX10', 'MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 841
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 374
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 683
				continue
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 458
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 891
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 374
				continue
			raise Exception('Invalid (classical) memory state at 654')
		if current_state == 1035:
			actions = ['RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			raise Exception('Invalid (classical) memory state at 1035')
		if current_state == 437:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 742
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 643
				continue
			raise Exception('Invalid (classical) memory state at 437')
		if current_state == 668:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 697
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 240
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 916
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 629
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 546
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 686
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 735
				continue
			raise Exception('Invalid (classical) memory state at 668')
		if current_state == 643:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 467
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 991
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 320
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 641
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 729
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 735
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 3:
				current_state = 654
				continue
			raise Exception('Invalid (classical) memory state at 643')
		if current_state == 618:
			actions = ['MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 678
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 456
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 553
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 470
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 841
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 365
				continue
			raise Exception('Invalid (classical) memory state at 618')
		if current_state == 637:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 462
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 595
				continue
			raise Exception('Invalid (classical) memory state at 637')
		if current_state == 543:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 729
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 808
				continue
			raise Exception('Invalid (classical) memory state at 543')
		if current_state == 616:
			actions = ['CX10', 'MEAS1', 'H0', 'H1', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 553
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 285
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 893
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 648
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 808
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			raise Exception('Invalid (classical) memory state at 616')
		if current_state == 390:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 831
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 518
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 761
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 728
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 447
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 1039
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 329
				continue
			raise Exception('Invalid (classical) memory state at 390')
		if current_state == 605:
			actions = ['CX10', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 518
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 417
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 449
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 763
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 3:
				current_state = 654
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 991
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 275
				continue
			raise Exception('Invalid (classical) memory state at 605')
		if current_state == 673:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 443
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 417
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 831
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 566
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 753
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 197
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 1039
				continue
			raise Exception('Invalid (classical) memory state at 673')
		if current_state == 582:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 605
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 743
				continue
			raise Exception('Invalid (classical) memory state at 582')
		if current_state == 968:
			actions = ['RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			raise Exception('Invalid (classical) memory state at 968')
		if current_state == 1039:
			actions = ['RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 0:
				current_state = 569
				continue
			raise Exception('Invalid (classical) memory state at 1039')
		if current_state == 950:
			actions = ['RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 3:
				current_state = 654
				continue
			raise Exception('Invalid (classical) memory state at 950')
		if current_state == 490:
			actions = ['CX10', 'MEAS1', 'H0', 'H1', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 657
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 962
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 231
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 672
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 502
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 0:
				current_state = 569
				continue
			raise Exception('Invalid (classical) memory state at 490')
		if current_state == 317:
			actions = ['NOTEntangled']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == NOTEntangled and simulator.meas_cache.get_memory_val() == 11:
				current_state = 62
				continue
			raise Exception('Invalid (classical) memory state at 317')
		if current_state == 473:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 747
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 612
				continue
			raise Exception('Invalid (classical) memory state at 473')
		if current_state == 420:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 761
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 666
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 640
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 642
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 791
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 968
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 208
				continue
			raise Exception('Invalid (classical) memory state at 420')
		if current_state == 512:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 785
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 707
				continue
			raise Exception('Invalid (classical) memory state at 512')
		if current_state == 208:
			actions = ['NOTEntangled']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == NOTEntangled and simulator.meas_cache.get_memory_val() == 8:
				current_state = 30
				continue
			raise Exception('Invalid (classical) memory state at 208')
		if current_state == 682:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 635
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 950
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 291
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 670
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 761
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 751
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 291
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 1039
				continue
			raise Exception('Invalid (classical) memory state at 682')
		if current_state == 451:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 635
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 790
				continue
			raise Exception('Invalid (classical) memory state at 451')
		if current_state == 588:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 550
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 1027
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 374
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 454
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 633
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 635
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			raise Exception('Invalid (classical) memory state at 588')
		if current_state == 818:
			actions = ['H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 531
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 580
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 680
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 291
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 893
				continue
			raise Exception('Invalid (classical) memory state at 818')
		if current_state == 717:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 518
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 531
				continue
			raise Exception('Invalid (classical) memory state at 717')
		if current_state == 695:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 559
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 515
				continue
			raise Exception('Invalid (classical) memory state at 695')
		if current_state == 773:
			actions = ['CX10', 'MEAS1', 'H0', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 580
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 317
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 891
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 684
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 480
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 612
				continue
			raise Exception('Invalid (classical) memory state at 773')
		if current_state == 570:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 502
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 551
				continue
			raise Exception('Invalid (classical) memory state at 570')
		if current_state == 443:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 673
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 785
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 731
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 748
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 706
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 968
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 208
				continue
			raise Exception('Invalid (classical) memory state at 443')
		if current_state == 488:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 666
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 465
				continue
			raise Exception('Invalid (classical) memory state at 488')
		if current_state == 502:
			actions = ['CX10', 'MEAS1', 'H0', 'H1', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 476
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 285
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 893
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 414
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 490
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 0:
				current_state = 569
				continue
			raise Exception('Invalid (classical) memory state at 502')
		if current_state == 566:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 499
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 950
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 291
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 826
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 673
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 825
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 291
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 1039
				continue
			raise Exception('Invalid (classical) memory state at 566')
		if current_state == 762:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 578
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 499
				continue
			raise Exception('Invalid (classical) memory state at 762')
		if current_state == 633:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 829
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 991
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 320
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 714
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 588
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 499
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			raise Exception('Invalid (classical) memory state at 633')
		if current_state == 613:
			actions = ['CX10', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 448
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 557
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 698
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 740
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 991
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 275
				continue
			raise Exception('Invalid (classical) memory state at 613')
		if current_state == 697:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 668
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 285
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 916
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 431
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 557
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 760
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 1040
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 285
				continue
			raise Exception('Invalid (classical) memory state at 697')
		if current_state == 797:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 557
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 507
				continue
			raise Exception('Invalid (classical) memory state at 797')
		if current_state == 546:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 402
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 1035
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 332
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 389
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 668
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 772
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 732
				continue
			raise Exception('Invalid (classical) memory state at 546')
		if current_state == 450:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 453
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 563
				continue
			raise Exception('Invalid (classical) memory state at 450')
		if current_state == 496:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 634
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 523
				continue
			raise Exception('Invalid (classical) memory state at 496')
		if current_state == 634:
			actions = ['H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 614
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 476
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 608
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 0:
				current_state = 569
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 291
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 893
				continue
			raise Exception('Invalid (classical) memory state at 634')
		if current_state == 779:
			actions = ['MEAS0', 'CX01', 'CX10', 'MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 962
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 302
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 448
				continue
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 772
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 962
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 221
				continue
			raise Exception('Invalid (classical) memory state at 779')
		if current_state == 522:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 711
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 448
				continue
			raise Exception('Invalid (classical) memory state at 522')
		if current_state == 594:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 463
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 240
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 916
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 599
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 425
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 814
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 554
				continue
			raise Exception('Invalid (classical) memory state at 594')
		if current_state == 802:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 735
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 616
				continue
			raise Exception('Invalid (classical) memory state at 802')
		if current_state == 709:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 469
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 557
				continue
			raise Exception('Invalid (classical) memory state at 709')
		if current_state == 469:
			actions = ['H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 711
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 803
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 606
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 0:
				current_state = 569
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 1027
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 380
				continue
			raise Exception('Invalid (classical) memory state at 469')
		if current_state == 693:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 563
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 532
				continue
			raise Exception('Invalid (classical) memory state at 693')
		if current_state == 795:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 698
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 916
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 225
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 719
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 772
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 425
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 415
				continue
			raise Exception('Invalid (classical) memory state at 795')
		if current_state == 563:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 603
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 291
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 991
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 611
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 554
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 415
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 3:
				current_state = 654
				continue
			raise Exception('Invalid (classical) memory state at 563')
		if current_state == 732:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 491
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 291
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 991
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 401
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 735
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 545
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 3:
				current_state = 654
				continue
			raise Exception('Invalid (classical) memory state at 732')
		if current_state == 532:
			actions = ['CX10', 'MEAS1', 'H0', 'H1', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 822
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 380
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 962
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 564
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 775
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			raise Exception('Invalid (classical) memory state at 532')
		if current_state == 792:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 729
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 808
				continue
			raise Exception('Invalid (classical) memory state at 792')
		if current_state == 729:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 449
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 1027
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 374
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 823
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 643
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 554
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 3:
				current_state = 654
				continue
			raise Exception('Invalid (classical) memory state at 729')
		if current_state == 657:
			actions = ['MEAS0', 'CX01', 'CX10', 'MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 968
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 231
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 614
				continue
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 490
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 1035
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 231
				continue
			raise Exception('Invalid (classical) memory state at 657')
		if current_state == 460:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 614
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 817
				continue
			raise Exception('Invalid (classical) memory state at 460')
		if current_state == 683:
			actions = ['MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 531
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 445
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 654
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 587
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 3:
				current_state = 654
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 841
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 302
				continue
			raise Exception('Invalid (classical) memory state at 683')
		if current_state == 458:
			actions = ['CX10', 'MEAS1', 'H0', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 654
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 891
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 195
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 824
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 498
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 3:
				current_state = 654
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 808
				continue
			raise Exception('Invalid (classical) memory state at 458')
		if current_state == 742:
			actions = ['CX10', 'MEAS1', 'H0', 'H1', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 756
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 962
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 231
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 674
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 529
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			raise Exception('Invalid (classical) memory state at 742')
		if current_state == 629:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 735
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 529
				continue
			raise Exception('Invalid (classical) memory state at 629')
		if current_state == 735:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 658
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 1027
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 204
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 475
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 732
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 643
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 3:
				current_state = 654
				continue
			raise Exception('Invalid (classical) memory state at 735')
		if current_state == 467:
			actions = ['MEAS0', 'CX01', 'CX10', 'MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 962
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 302
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 518
				continue
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 643
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 950
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 320
				continue
			raise Exception('Invalid (classical) memory state at 467')
		if current_state == 641:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 742
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 643
				continue
			raise Exception('Invalid (classical) memory state at 641')
		if current_state == 553:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 616
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 240
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 893
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 646
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 618
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 738
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 841
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 204
				continue
			raise Exception('Invalid (classical) memory state at 553')
		if current_state == 470:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 523
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 678
				continue
			raise Exception('Invalid (classical) memory state at 470')
		if current_state == 648:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 554
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 616
				continue
			raise Exception('Invalid (classical) memory state at 648')
		if current_state == 518:
			actions = ['CX10', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 605
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 534
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 467
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 725
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 3:
				current_state = 654
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 962
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 374
				continue
			raise Exception('Invalid (classical) memory state at 518')
		if current_state == 761:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 420
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 534
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 390
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 682
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 623
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 197
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 1039
				continue
			raise Exception('Invalid (classical) memory state at 761')
		if current_state == 447:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 518
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 531
				continue
			raise Exception('Invalid (classical) memory state at 447')
		if current_state == 417:
			actions = ['CX10', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 785
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 605
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 491
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 544
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 3:
				current_state = 654
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 204
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 962
				continue
			raise Exception('Invalid (classical) memory state at 417')
		if current_state == 449:
			actions = ['MEAS0', 'CX01', 'CX10', 'MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 1040
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 275
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 605
				continue
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 729
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 1027
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 195
				continue
			raise Exception('Invalid (classical) memory state at 449')
		if current_state == 763:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 605
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 743
				continue
			raise Exception('Invalid (classical) memory state at 763')
		if current_state == 753:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 798
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 417
				continue
			raise Exception('Invalid (classical) memory state at 753')
		if current_state == 743:
			actions = ['H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 798
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 413
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 444
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 1027
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 231
				continue
			raise Exception('Invalid (classical) memory state at 743')
		if current_state == 672:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 490
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 619
				continue
			raise Exception('Invalid (classical) memory state at 672')
		if current_state == 666:
			actions = ['CX10', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 534
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 830
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 591
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 530
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 3:
				current_state = 654
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 991
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 285
				continue
			raise Exception('Invalid (classical) memory state at 666')
		if current_state == 640:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 731
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 830
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 420
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 829
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 664
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 1039
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 329
				continue
			raise Exception('Invalid (classical) memory state at 640')
		if current_state == 791:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 707
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 666
				continue
			raise Exception('Invalid (classical) memory state at 791')
		if current_state == 785:
			actions = ['CX10', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 417
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 620
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 658
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 494
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 3:
				current_state = 654
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 991
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 285
				continue
			raise Exception('Invalid (classical) memory state at 785')
		if current_state == 707:
			actions = ['H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 787
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 821
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 690
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 1027
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 380
				continue
			raise Exception('Invalid (classical) memory state at 707')
		if current_state == 670:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 499
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 790
				continue
			raise Exception('Invalid (classical) memory state at 670')
		if current_state == 751:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 417
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 818
				continue
			raise Exception('Invalid (classical) memory state at 751')
		if current_state == 790:
			actions = ['CX10', 'MEAS1', 'H0', 'H1', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 803
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 380
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 962
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 407
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 811
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 0:
				current_state = 569
				continue
			raise Exception('Invalid (classical) memory state at 790')
		if current_state == 550:
			actions = ['MEAS0', 'CX01', 'CX10', 'MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 1027
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 352
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 731
				continue
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 588
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 1027
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 195
				continue
			raise Exception('Invalid (classical) memory state at 550')
		if current_state == 454:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 811
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 588
				continue
			raise Exception('Invalid (classical) memory state at 454')
		if current_state == 580:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 773
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 204
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 891
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 489
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 818
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 801
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 893
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 197
				continue
			raise Exception('Invalid (classical) memory state at 580')
		if current_state == 680:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 534
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 818
				continue
			raise Exception('Invalid (classical) memory state at 680')
		if current_state == 684:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 551
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 612
				continue
			raise Exception('Invalid (classical) memory state at 684')
		if current_state == 231:
			actions = ['Entangled']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == Entangled and simulator.meas_cache.get_memory_val() == 4:
				current_state = 125
				continue
			raise Exception('Invalid (classical) memory state at 231')
		if current_state == 731:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 640
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 620
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 443
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 550
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 430
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 968
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 352
				continue
			raise Exception('Invalid (classical) memory state at 731')
		if current_state == 706:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 785
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 465
				continue
			raise Exception('Invalid (classical) memory state at 706')
		if current_state == 465:
			actions = ['H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 636
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 688
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 675
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 1027
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 380
				continue
			raise Exception('Invalid (classical) memory state at 465')
		if current_state == 476:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 502
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 240
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 893
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 428
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 634
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 452
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 0:
				current_state = 569
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 893
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 197
				continue
			raise Exception('Invalid (classical) memory state at 476')
		if current_state == 414:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 747
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 502
				continue
			raise Exception('Invalid (classical) memory state at 414')
		if current_state == 826:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 635
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 578
				continue
			raise Exception('Invalid (classical) memory state at 826')
		if current_state == 825:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 798
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 534
				continue
			raise Exception('Invalid (classical) memory state at 825')
		if current_state == 578:
			actions = ['CX10', 'MEAS1', 'H0', 'H1', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 521
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 380
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 962
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 745
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 628
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 0:
				current_state = 569
				continue
			raise Exception('Invalid (classical) memory state at 578')
		if current_state == 829:
			actions = ['MEAS0', 'CX01', 'CX10', 'MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 1039
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 320
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 640
				continue
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 633
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 950
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 320
				continue
			raise Exception('Invalid (classical) memory state at 829')
		if current_state == 714:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 633
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 628
				continue
			raise Exception('Invalid (classical) memory state at 714')
		if current_state == 698:
			actions = ['MEAS0', 'CX01', 'CX10', 'MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 1040
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 275
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 613
				continue
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 795
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 916
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 275
				continue
			raise Exception('Invalid (classical) memory state at 698')
		if current_state == 740:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 613
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 517
				continue
			raise Exception('Invalid (classical) memory state at 740')
		if current_state == 431:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 554
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 529
				continue
			raise Exception('Invalid (classical) memory state at 431')
		if current_state == 760:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 507
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 514
				continue
			raise Exception('Invalid (classical) memory state at 760')
		if current_state == 507:
			actions = ['H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 517
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 521
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 403
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 0:
				current_state = 569
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 1027
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 380
				continue
			raise Exception('Invalid (classical) memory state at 507')
		if current_state == 389:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 732
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 453
				continue
			raise Exception('Invalid (classical) memory state at 389')
		if current_state == 772:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 779
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 1035
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 221
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 694
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 795
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 546
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 545
				continue
			raise Exception('Invalid (classical) memory state at 772')
		if current_state == 453:
			actions = ['CX10', 'MEAS1', 'H0', 'H1', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 555
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 380
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 962
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 663
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 572
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			raise Exception('Invalid (classical) memory state at 453')
		if current_state == 608:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 602
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 634
				continue
			raise Exception('Invalid (classical) memory state at 608')
		if current_state == 711:
			actions = ['H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 469
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 687
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 813
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 0:
				current_state = 569
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 893
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 320
				continue
			raise Exception('Invalid (classical) memory state at 711')
		if current_state == 599:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 554
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 616
				continue
			raise Exception('Invalid (classical) memory state at 599')
		if current_state == 554:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 591
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 1027
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 204
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 386
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 563
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 729
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 3:
				current_state = 654
				continue
			raise Exception('Invalid (classical) memory state at 554')
		if current_state == 803:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 790
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 380
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 1035
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 387
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 469
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 396
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 0:
				current_state = 569
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 968
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 380
				continue
			raise Exception('Invalid (classical) memory state at 803')
		if current_state == 606:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 469
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 514
				continue
			raise Exception('Invalid (classical) memory state at 606')
		if current_state == 380:
			actions = ['NOTEntangled']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == NOTEntangled and simulator.meas_cache.get_memory_val() == 8:
				current_state = 8
				continue
			raise Exception('Invalid (classical) memory state at 380')
		if current_state == 719:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 415
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 775
				continue
			raise Exception('Invalid (classical) memory state at 719')
		if current_state == 415:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 768
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 1027
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 374
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 701
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 545
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 563
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 3:
				current_state = 654
				continue
			raise Exception('Invalid (classical) memory state at 415')
		if current_state == 603:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 563
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 950
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 291
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 601
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 534
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 482
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 3:
				current_state = 654
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 365
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 962
				continue
			raise Exception('Invalid (classical) memory state at 603')
		if current_state == 611:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 563
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 532
				continue
			raise Exception('Invalid (classical) memory state at 611')
		if current_state == 491:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 732
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 950
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 291
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 662
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 417
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 441
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 3:
				current_state = 654
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 365
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 962
				continue
			raise Exception('Invalid (classical) memory state at 491')
		if current_state == 401:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 732
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 453
				continue
			raise Exception('Invalid (classical) memory state at 401')
		if current_state == 545:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 549
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 991
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 320
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 656
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 415
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 732
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 3:
				current_state = 654
				continue
			raise Exception('Invalid (classical) memory state at 545')
		if current_state == 822:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 532
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 380
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 1035
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 624
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 589
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 446
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 991
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 334
				continue
			raise Exception('Invalid (classical) memory state at 822')
		if current_state == 564:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 563
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 532
				continue
			raise Exception('Invalid (classical) memory state at 564')
		if current_state == 775:
			actions = ['CX10', 'MEAS1', 'H0', 'H1', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 500
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 893
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 275
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 726
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 532
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			raise Exception('Invalid (classical) memory state at 775')
		if current_state == 823:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 729
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 808
				continue
			raise Exception('Invalid (classical) memory state at 823')
		if current_state == 445:
			actions = ['MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 818
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 683
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 427
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 519
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 3:
				current_state = 654
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 841
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 365
				continue
			raise Exception('Invalid (classical) memory state at 445')
		if current_state == 587:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 518
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 531
				continue
			raise Exception('Invalid (classical) memory state at 587')
		if current_state == 824:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 729
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 808
				continue
			raise Exception('Invalid (classical) memory state at 824')
		if current_state == 498:
			actions = ['CX10', 'MEAS1', 'H0', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 427
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 317
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 891
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 538
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 458
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 3:
				current_state = 654
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 616
				continue
			raise Exception('Invalid (classical) memory state at 498')
		if current_state == 756:
			actions = ['MEAS0', 'CX01', 'CX10', 'MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 991
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 316
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 513
				continue
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 742
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 1035
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 231
				continue
			raise Exception('Invalid (classical) memory state at 756')
		if current_state == 674:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 742
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 643
				continue
			raise Exception('Invalid (classical) memory state at 674')
		if current_state == 529:
			actions = ['CX10', 'MEAS1', 'H0', 'H1', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 598
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 285
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 893
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 388
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 742
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			raise Exception('Invalid (classical) memory state at 529')
		if current_state == 658:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 735
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 1027
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 317
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 574
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 785
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 548
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 3:
				current_state = 654
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 1040
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 285
				continue
			raise Exception('Invalid (classical) memory state at 658')
		if current_state == 475:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 735
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 529
				continue
			raise Exception('Invalid (classical) memory state at 475')
		if current_state == 646:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 735
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 616
				continue
			raise Exception('Invalid (classical) memory state at 646')
		if current_state == 738:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 602
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 678
				continue
			raise Exception('Invalid (classical) memory state at 738')
		if current_state == 534:
			actions = ['CX10', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 666
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 518
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 603
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 495
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 3:
				current_state = 654
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 204
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 962
				continue
			raise Exception('Invalid (classical) memory state at 534')
		if current_state == 725:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 518
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 531
				continue
			raise Exception('Invalid (classical) memory state at 725')
		if current_state == 623:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 534
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 818
				continue
			raise Exception('Invalid (classical) memory state at 623')
		if current_state == 544:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 798
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 417
				continue
			raise Exception('Invalid (classical) memory state at 544')
		if current_state == 798:
			actions = ['H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 743
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 486
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 685
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 291
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 893
				continue
			raise Exception('Invalid (classical) memory state at 798')
		if current_state == 413:
			actions = ['MEAS0', 'CX01', 'CX10', 'MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 968
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 231
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 743
				continue
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 647
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 991
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 199
				continue
			raise Exception('Invalid (classical) memory state at 413')
		if current_state == 444:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 605
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 743
				continue
			raise Exception('Invalid (classical) memory state at 444')
		if current_state == 830:
			actions = ['CX10', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 620
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 666
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 549
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 527
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 3:
				current_state = 654
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 962
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 374
				continue
			raise Exception('Invalid (classical) memory state at 830')
		if current_state == 591:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 554
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 1027
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 317
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 576
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 666
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 669
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 3:
				current_state = 654
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 1040
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 285
				continue
			raise Exception('Invalid (classical) memory state at 591')
		if current_state == 530:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 707
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 666
				continue
			raise Exception('Invalid (classical) memory state at 530')
		if current_state == 664:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 787
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 830
				continue
			raise Exception('Invalid (classical) memory state at 664')
		if current_state == 620:
			actions = ['CX10', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 830
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 785
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 768
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 661
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 3:
				current_state = 654
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 991
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 275
				continue
			raise Exception('Invalid (classical) memory state at 620')
		if current_state == 494:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 785
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 465
				continue
			raise Exception('Invalid (classical) memory state at 494')
		if current_state == 787:
			actions = ['H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 707
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 466
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 567
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 893
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 320
				continue
			raise Exception('Invalid (classical) memory state at 787')
		if current_state == 821:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 592
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 991
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 282
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 579
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 707
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 799
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 968
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 380
				continue
			raise Exception('Invalid (classical) memory state at 821')
		if current_state == 690:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 707
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 666
				continue
			raise Exception('Invalid (classical) memory state at 690')
		if current_state == 407:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 635
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 790
				continue
			raise Exception('Invalid (classical) memory state at 407')
		if current_state == 811:
			actions = ['CX10', 'MEAS1', 'H0', 'H1', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 687
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 893
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 275
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 659
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 790
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 0:
				current_state = 569
				continue
			raise Exception('Invalid (classical) memory state at 811')
		if current_state == 489:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 747
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 612
				continue
			raise Exception('Invalid (classical) memory state at 489')
		if current_state == 801:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 417
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 818
				continue
			raise Exception('Invalid (classical) memory state at 801')
		if current_state == 430:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 636
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 620
				continue
			raise Exception('Invalid (classical) memory state at 430')
		if current_state == 636:
			actions = ['H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 465
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 501
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 755
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 1027
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 231
				continue
			raise Exception('Invalid (classical) memory state at 636')
		if current_state == 688:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 393
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 991
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 282
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 581
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 465
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 436
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 968
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 380
				continue
			raise Exception('Invalid (classical) memory state at 688')
		if current_state == 675:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 785
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 465
				continue
			raise Exception('Invalid (classical) memory state at 675')
		if current_state == 428:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 502
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 551
				continue
			raise Exception('Invalid (classical) memory state at 428')
		if current_state == 452:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 634
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 523
				continue
			raise Exception('Invalid (classical) memory state at 452')
		if current_state == 521:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 578
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 380
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 1035
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 689
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 507
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 552
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 0:
				current_state = 569
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 968
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 380
				continue
			raise Exception('Invalid (classical) memory state at 521')
		if current_state == 745:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 578
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 499
				continue
			raise Exception('Invalid (classical) memory state at 745')
		if current_state == 628:
			actions = ['CX10', 'MEAS1', 'H0', 'H1', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 434
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 962
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 231
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 700
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 578
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 0:
				current_state = 569
				continue
			raise Exception('Invalid (classical) memory state at 628')
		if current_state == 517:
			actions = ['H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 507
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 434
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 715
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 0:
				current_state = 569
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 1027
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 231
				continue
			raise Exception('Invalid (classical) memory state at 517')
		if current_state == 403:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 557
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 507
				continue
			raise Exception('Invalid (classical) memory state at 403')
		if current_state == 694:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 545
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 572
				continue
			raise Exception('Invalid (classical) memory state at 694')
		if current_state == 555:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 453
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 380
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 1035
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 667
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 639
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 741
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 991
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 334
				continue
			raise Exception('Invalid (classical) memory state at 555')
		if current_state == 663:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 732
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 453
				continue
			raise Exception('Invalid (classical) memory state at 663')
		if current_state == 572:
			actions = ['CX10', 'MEAS1', 'H0', 'H1', 'RESET']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 702
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 962
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 231
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 568
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 453
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			raise Exception('Invalid (classical) memory state at 572')
		if current_state == 687:
			actions = ['MEAS0', 'CX01', 'CX10', 'MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 893
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 329
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 711
				continue
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 811
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 893
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 225
				continue
			raise Exception('Invalid (classical) memory state at 687')
		if current_state == 813:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 711
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 448
				continue
			raise Exception('Invalid (classical) memory state at 813')
		if current_state == 386:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 554
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 616
				continue
			raise Exception('Invalid (classical) memory state at 386')
		if current_state == 387:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 499
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 790
				continue
			raise Exception('Invalid (classical) memory state at 387')
		if current_state == 396:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 469
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 557
				continue
			raise Exception('Invalid (classical) memory state at 396')
		if current_state == 768:
			actions = ['MEAS0', 'CX01', 'CX10', 'MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 1040
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 275
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 620
				continue
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 415
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 1027
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 195
				continue
			raise Exception('Invalid (classical) memory state at 768')
		if current_state == 701:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 415
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 775
				continue
			raise Exception('Invalid (classical) memory state at 701')
		if current_state == 601:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 732
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 532
				continue
			raise Exception('Invalid (classical) memory state at 601')
		if current_state == 482:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 417
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 818
				continue
			raise Exception('Invalid (classical) memory state at 482')
		if current_state == 662:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 453
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 563
				continue
			raise Exception('Invalid (classical) memory state at 662')
		if current_state == 441:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 798
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 534
				continue
			raise Exception('Invalid (classical) memory state at 441')
		if current_state == 549:
			actions = ['MEAS0', 'CX01', 'CX10', 'MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 962
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 302
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 830
				continue
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 545
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 950
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 320
				continue
			raise Exception('Invalid (classical) memory state at 549')
		if current_state == 656:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 545
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 572
				continue
			raise Exception('Invalid (classical) memory state at 656')
		if current_state == 624:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 732
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 532
				continue
			raise Exception('Invalid (classical) memory state at 624')
		if current_state == 589:
			actions = ['MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 469
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 509
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 822
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 749
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 1040
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 334
				continue
			raise Exception('Invalid (classical) memory state at 589')
		if current_state == 446:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 469
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 557
				continue
			raise Exception('Invalid (classical) memory state at 446')
		if current_state == 334:
			actions = ['NOTEntangled']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == NOTEntangled and simulator.meas_cache.get_memory_val() == 10:
				current_state = 165
				continue
			raise Exception('Invalid (classical) memory state at 334')
		if current_state == 500:
			actions = ['MEAS0', 'CX01', 'CX10', 'MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 841
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 374
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 509
				continue
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 775
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 893
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 225
				continue
			raise Exception('Invalid (classical) memory state at 500')
		if current_state == 726:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 415
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 775
				continue
			raise Exception('Invalid (classical) memory state at 726')
		if current_state == 427:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 498
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 204
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 891
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 782
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 445
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 805
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 3:
				current_state = 654
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 841
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 204
				continue
			raise Exception('Invalid (classical) memory state at 427')
		if current_state == 519:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 534
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 818
				continue
			raise Exception('Invalid (classical) memory state at 519')
		if current_state == 538:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 554
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 616
				continue
			raise Exception('Invalid (classical) memory state at 538')
		if current_state == 513:
			actions = ['MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 614
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 478
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 756
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 625
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 1040
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 316
				continue
			raise Exception('Invalid (classical) memory state at 513')
		if current_state == 598:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 529
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 240
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 893
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 505
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 478
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 752
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 841
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 204
				continue
			raise Exception('Invalid (classical) memory state at 598')
		if current_state == 388:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 735
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 529
				continue
			raise Exception('Invalid (classical) memory state at 388')
		if current_state == 574:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 554
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 529
				continue
			raise Exception('Invalid (classical) memory state at 574')
		if current_state == 548:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 666
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 465
				continue
			raise Exception('Invalid (classical) memory state at 548')
		if current_state == 495:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 534
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 818
				continue
			raise Exception('Invalid (classical) memory state at 495')
		if current_state == 486:
			actions = ['CX10', 'MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 718
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 204
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 891
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 439
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 798
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 810
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 893
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 197
				continue
			raise Exception('Invalid (classical) memory state at 486')
		if current_state == 685:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 798
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 417
				continue
			raise Exception('Invalid (classical) memory state at 685')
		if current_state == 647:
			actions = ['CX10', 'MEAS1', 'H0', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 413
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 950
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 199
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 556
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 718
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 490
				continue
			raise Exception('Invalid (classical) memory state at 647')
		if current_state == 527:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 787
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 830
				continue
			raise Exception('Invalid (classical) memory state at 527')
		if current_state == 576:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 735
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 616
				continue
			raise Exception('Invalid (classical) memory state at 576')
		if current_state == 669:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 785
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 707
				continue
			raise Exception('Invalid (classical) memory state at 669')
		if current_state == 661:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 636
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 620
				continue
			raise Exception('Invalid (classical) memory state at 661')
		if current_state == 466:
			actions = ['MEAS0', 'CX01', 'CX10', 'MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 893
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 329
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 787
				continue
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 575
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 891
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 374
				continue
			raise Exception('Invalid (classical) memory state at 466')
		if current_state == 567:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 787
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 830
				continue
			raise Exception('Invalid (classical) memory state at 567')
		if current_state == 592:
			actions = ['CX10', 'MEAS1', 'H0', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 821
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 950
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 282
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 585
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 575
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 790
				continue
			raise Exception('Invalid (classical) memory state at 592')
		if current_state == 282:
			actions = ['NOTEntangled']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == NOTEntangled and simulator.meas_cache.get_memory_val() == 9:
				current_state = 70
				continue
			raise Exception('Invalid (classical) memory state at 282')
		if current_state == 579:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 499
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 790
				continue
			raise Exception('Invalid (classical) memory state at 579')
		if current_state == 799:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 785
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 707
				continue
			raise Exception('Invalid (classical) memory state at 799')
		if current_state == 659:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 811
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 588
				continue
			raise Exception('Invalid (classical) memory state at 659')
		if current_state == 316:
			actions = ['Entangled']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == Entangled and simulator.meas_cache.get_memory_val() == 6:
				current_state = 99
				continue
			raise Exception('Invalid (classical) memory state at 316')
		if current_state == 199:
			actions = ['Entangled']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == Entangled and simulator.meas_cache.get_memory_val() == 5:
				current_state = 167
				continue
			raise Exception('Invalid (classical) memory state at 199')
		if current_state == 501:
			actions = ['MEAS0', 'CX01', 'CX10', 'MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 968
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 231
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 636
				continue
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 771
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 991
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 199
				continue
			raise Exception('Invalid (classical) memory state at 501')
		if current_state == 755:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 636
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 620
				continue
			raise Exception('Invalid (classical) memory state at 755')
		if current_state == 393:
			actions = ['CX10', 'MEAS1', 'H0', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 688
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 950
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 282
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 681
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 771
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 578
				continue
			raise Exception('Invalid (classical) memory state at 393')
		if current_state == 581:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 635
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 578
				continue
			raise Exception('Invalid (classical) memory state at 581')
		if current_state == 436:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 666
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 465
				continue
			raise Exception('Invalid (classical) memory state at 436')
		if current_state == 689:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 635
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 578
				continue
			raise Exception('Invalid (classical) memory state at 689')
		if current_state == 552:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 507
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 514
				continue
			raise Exception('Invalid (classical) memory state at 552')
		if current_state == 434:
			actions = ['MEAS0', 'CX01', 'CX10', 'MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 968
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 231
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 517
				continue
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 628
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 1035
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 231
				continue
			raise Exception('Invalid (classical) memory state at 434')
		if current_state == 700:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 633
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 628
				continue
			raise Exception('Invalid (classical) memory state at 700')
		if current_state == 715:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 613
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 517
				continue
			raise Exception('Invalid (classical) memory state at 715')
		if current_state == 667:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 453
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 563
				continue
			raise Exception('Invalid (classical) memory state at 667')
		if current_state == 639:
			actions = ['MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 507
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 794
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 555
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 724
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 1040
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 334
				continue
			raise Exception('Invalid (classical) memory state at 639')
		if current_state == 741:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 507
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 514
				continue
			raise Exception('Invalid (classical) memory state at 741')
		if current_state == 702:
			actions = ['MEAS0', 'CX01', 'CX10', 'MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 991
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 316
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 794
				continue
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 572
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 1035
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 231
				continue
			raise Exception('Invalid (classical) memory state at 702')
		if current_state == 568:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 545
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 572
				continue
			raise Exception('Invalid (classical) memory state at 568')
		if current_state == 509:
			actions = ['MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 711
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 589
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 500
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 562
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 841
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 302
				continue
			raise Exception('Invalid (classical) memory state at 509')
		if current_state == 749:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 469
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 514
				continue
			raise Exception('Invalid (classical) memory state at 749')
		if current_state == 782:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 735
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 616
				continue
			raise Exception('Invalid (classical) memory state at 782')
		if current_state == 805:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 417
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 818
				continue
			raise Exception('Invalid (classical) memory state at 805')
		if current_state == 478:
			actions = ['MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 634
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 513
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 598
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 583
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 841
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 365
				continue
			raise Exception('Invalid (classical) memory state at 478')
		if current_state == 625:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 614
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 817
				continue
			raise Exception('Invalid (classical) memory state at 625')
		if current_state == 505:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 554
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 529
				continue
			raise Exception('Invalid (classical) memory state at 505')
		if current_state == 752:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 634
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 523
				continue
			raise Exception('Invalid (classical) memory state at 752')
		if current_state == 718:
			actions = ['CX10', 'MEAS1', 'H0', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 486
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 317
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 891
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 400
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 647
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 502
				continue
			raise Exception('Invalid (classical) memory state at 718')
		if current_state == 439:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 502
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 551
				continue
			raise Exception('Invalid (classical) memory state at 439')
		if current_state == 810:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 798
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 534
				continue
			raise Exception('Invalid (classical) memory state at 810')
		if current_state == 556:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 490
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 619
				continue
			raise Exception('Invalid (classical) memory state at 556')
		if current_state == 575:
			actions = ['CX10', 'MEAS1', 'H0', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 466
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 891
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 195
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 660
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 592
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 811
				continue
			raise Exception('Invalid (classical) memory state at 575')
		if current_state == 585:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 635
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 790
				continue
			raise Exception('Invalid (classical) memory state at 585')
		if current_state == 771:
			actions = ['CX10', 'MEAS1', 'H0', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == CX10 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 501
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 950
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 199
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 766
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 393
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 1:
				current_state = 807
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 628
				continue
			raise Exception('Invalid (classical) memory state at 771')
		if current_state == 681:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 578
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 499
				continue
			raise Exception('Invalid (classical) memory state at 681')
		if current_state == 794:
			actions = ['MEAS1', 'H0', 'CX01', 'H1', 'RESET', 'MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 517
				continue
			if choosen_action == H0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 639
				continue
			if choosen_action == CX01 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 702
				continue
			if choosen_action == H1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 784
				continue
			if choosen_action == RESET and simulator.meas_cache.get_memory_val() == 2:
				current_state = 828
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 3:
				current_state = 1040
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 316
				continue
			raise Exception('Invalid (classical) memory state at 794')
		if current_state == 724:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 557
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 507
				continue
			raise Exception('Invalid (classical) memory state at 724')
		if current_state == 562:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 711
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 448
				continue
			raise Exception('Invalid (classical) memory state at 562')
		if current_state == 583:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 602
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 634
				continue
			raise Exception('Invalid (classical) memory state at 583')
		if current_state == 400:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 747
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 502
				continue
			raise Exception('Invalid (classical) memory state at 400')
		if current_state == 660:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 811
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 588
				continue
			raise Exception('Invalid (classical) memory state at 660')
		if current_state == 766:
			actions = ['MEAS0']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 1:
				current_state = 633
				continue
			if choosen_action == MEAS0 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 628
				continue
			raise Exception('Invalid (classical) memory state at 766')
		if current_state == 784:
			actions = ['MEAS1']
			choosen_action = random.choice(actions)
			simulator.apply_instructions(choosen_action)
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 2:
				current_state = 613
				continue
			if choosen_action == MEAS1 and simulator.meas_cache.get_memory_val() == 0:
				current_state = 517
				continue
			raise Exception('Invalid (classical) memory state at 784')
