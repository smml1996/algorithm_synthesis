from cmath import isclose
from typing import List, Optional
from cmemory import ClassicalState, cread, cwrite
from qmemory import QuantumState, get_qs_probability, handle_write
from qpu_utils import GateData, Op
from utils import Precision
import random
from ibm_noise_models import MeasChannel, NoiseModel, Instruction, QuantumChannel

class QSimulator:
    qmemory: QuantumState
    meas_cache: ClassicalState
    noise_model: NoiseModel
    count_executed_instructions: int
    log: List[str]
    def __init__(self, noise_model = NoiseModel(), seed=None, qubits_used=None, log=None) -> None:
        if log is None:
            self.log = []
        else:
            self.log = log
        if not (seed is None):
            self.log.append(f"SEED: {seed}")
            random.seed(seed)
        self.seed = seed
        self.noise_model = noise_model
        self.qmemory = QuantumState(0, qubits_used=qubits_used)
        self.meas_cache = ClassicalState()
        self.count_executed_instructions = 0        
        
    def apply_noise_model(self, instruction: Instruction):
        channel = self.noise_model.get_instruction_channel(instruction)
        
        if not instruction.is_meas_instruction():
            assert isinstance(channel, QuantumChannel)
            error_sequence = random.choices(channel.errors, weights=channel.probabilities, k=1)[0]
            
            for error_instruction in error_sequence:
                assert isinstance(error_instruction, GateData)
                self.qmemory = handle_write(self.qmemory, error_instruction)
        else:
            assert isinstance(channel, MeasChannel)
            current_memory_val = self.get_meas_cache_val(instruction.target)
            correct_prob = channel.get_ind_probability(current_memory_val, current_memory_val)
            wrong_val = (current_memory_val + 1) % 2
            wrong_prob = channel.get_ind_probability(current_memory_val, wrong_val)
            
            flip = random.choices([current_memory_val, wrong_val], weights=[correct_prob, wrong_prob], k=1)[0]
            
            if flip == 0:
                self.meas_cache = cwrite(self.meas_cache, Op.WRITE0, instruction.target)
            else:
                assert flip == 1
                self.meas_cache = cwrite(self.meas_cache, Op.WRITE1, instruction.target)

    def apply_instruction(self, instruction: Instruction, with_noise=True) -> Optional[int]:
        return_val = None
        if instruction.op in [Op.WRITE0, Op.WRITE1]:
            self.log.append(f"QPU: {instruction}")
            self.meas_cache = cwrite(self.meas_cache, instruction.op, instruction.target)
        elif not instruction.is_meas_instruction():
            self.log.append(f"QPU: {instruction}")
            self.qmemory = handle_write(self.qmemory, instruction.get_gate_data())
        else:
            projector0 = instruction.get_gate_data(True)
            projector1 = instruction.get_gate_data(False)
            prob0 = get_qs_probability(self.qmemory, instruction.target, is_zero=True)
            prob1 = 1.0 - prob0
            rel_tol = 1/(10**(Precision.PRECISION+1))
            assert isclose(prob0 + prob1, 1.0, rel_tol=rel_tol)
            projector = random.choices([projector0, projector1], [prob0, prob1], k=1)[0]
            self.log.append(f"QPU: {instruction} {projector.label}")
            self.qmemory = handle_write(self.qmemory, projector)
            if projector.label == Op.P0:
                self.meas_cache = cwrite(self.meas_cache, Op.WRITE0, instruction.target)
            else:
                assert projector.label == Op.P1
                self.meas_cache = cwrite(self.meas_cache, Op.WRITE1, instruction.target)
        
        self.count_executed_instructions += 1
        if with_noise:
            self.apply_noise_model(instruction)
        if instruction.is_meas_instruction():
            return_val = cread(self.meas_cache, instruction.target)
        return return_val
    
    def apply_instructions(self, instructions: List[Instruction]) -> Optional[int]:
        outcome = None
        for instruction in instructions:
            temp_outcome = self.apply_instruction(instruction)
            if temp_outcome != None:
                outcome = temp_outcome
        return outcome

    def get_meas_cache_val(self, address) -> bool:
        return cread(self.meas_cache, address)
