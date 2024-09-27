from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from qiskit.providers.fake_provider import *
from qiskit_aer.noise import NoiseModel as IBMNoiseModel
from qpu_utils import *
import json

from utils import Precision, invert_dict, myceil, myfloor


class HardwareSpec(Enum):
    # Quantum hardware names available in Qiskit
    TENERIFE = "fake_tenerife"
    JOHANNESBURG = "fake_johannesburg"
    PERTH = "fake_perth"
    LAGOS = "fake_lagos"
    NAIROBI = "fake_nairobi"
    HANOI = "fake_hanoi"
    CAIRO = "fake_cairo"
    MUMBAI = "fake_mumbai"
    KOLKATA = "fake_kolkata"
    PRAGUE = "fake_prague"
    ALMADEN = "fake_almaden"
    ARMONK = "fake_armonk"
    ATHENS = "fake_athens"
    AUCKLAND = "fake_auckland"
    BELEM = "fake_belem"
    BOEBLINGEN = "fake_boeblingen"
    BOGOTA = "fake_bogota"
    BROOKLYN = "fake_brooklyn"
    BURLINGTON = "fake_burlington"
    CAMBRIDGE = "fake_cambridge"
    CASABLANCA = "fake_casablanca"
    ESSEX = "fake_essex"
    GENEVA = "fake_geneva"
    GUADALUPE = "fake_guadalupe"
    LIMA = "fake_lima"
    LONDON = "fake_london"
    MANHATTAN = "fake_manhattan"
    MANILA = "fake_manila"
    MELBOURNE = "fake_melbourne"
    MONTREAL = "fake_montreal"
    OSLO = "fake_oslo"
    OURENSE = "fake_ourense"
    PARIS = "fake_paris"
    QUITO = "fake_quito"
    POUGHKEEPSIE = "fake_poughkeepsie"
    ROCHESTER = "fake_rochester"
    ROME = "fake_rome"
    RUESCHLIKON = "fake_rueschlikon"
    SANTIAGO = "fake_santiago"
    SINGAPORE = "fake_singapore"
    SYDNEY = "fake_sydney"
    TOKYO = "fake_tokyo"
    TORONTO = "fake_toronto"
    VIGO = "fake_vigo"
    WASHINGTON = "fake_washington"
    YORKTOWN = "fake_yorktown"
    JAKARTA = "fake_jakarta"
    def __repr__(self) -> str:
        return self.__str__()


def get_ibm_noise_model(hardware_spec: HardwareSpec, thermal_relaxation=True) -> NoiseModel:
    backend_ = hardware_spec
    if backend_ == HardwareSpec.TENERIFE:
        backend = FakeTenerife()
    elif backend_ == HardwareSpec.JOHANNESBURG:
        backend = FakeJohannesburg()
    elif backend_ == HardwareSpec.PERTH:
        backend = FakePerth()
    elif backend_ == HardwareSpec.LAGOS:
        backend = FakeLagos()
    elif backend_ == HardwareSpec.NAIROBI:
        backend = FakeNairobi()
    elif backend_ ==  HardwareSpec.HANOI:
        backend = FakeHanoi()
    elif backend_ == HardwareSpec.CAIRO:
        backend = FakeCairo()
    elif backend_ == HardwareSpec.MUMBAI:
        backend = FakeMumbai()
    elif backend_ == HardwareSpec.KOLKATA:
        backend = FakeKolkata()
    elif backend_ == HardwareSpec.PRAGUE:
        backend = FakePrague()
    elif backend_ == HardwareSpec.ALMADEN:
        backend = FakeAlmaden()
    elif backend_ == HardwareSpec.ARMONK:
        backend = FakeArmonk()
    elif backend_ == HardwareSpec.ATHENS:
        backend = FakeAthens()
    elif backend_ == HardwareSpec.AUCKLAND:
        backend = FakeAuckland()
    elif backend_ == HardwareSpec.BELEM:
        backend = FakeBelem()
    elif backend_ == HardwareSpec.BOEBLINGEN:
        backend = FakeBoeblingen()
    elif backend_ == HardwareSpec.BOGOTA:
        backend = FakeBogota()
    elif backend_ == HardwareSpec.BROOKLYN:
        backend = FakeBrooklyn()
    elif backend_ == HardwareSpec.BURLINGTON:
        backend = FakeBurlington()
    elif backend_ == HardwareSpec.CAMBRIDGE:
        backend = FakeCambridge()
    elif backend_ == HardwareSpec.CASABLANCA:
        backend = FakeCasablanca()
    elif backend_ == HardwareSpec.ESSEX:
        backend = FakeEssex()
    elif backend_ == HardwareSpec.GENEVA:
        backend = FakeGeneva()
    elif backend_ == HardwareSpec.GUADALUPE:
        backend = FakeGuadalupe()
    elif backend_ == HardwareSpec.LIMA:
        backend = FakeLima()
    elif backend_ == HardwareSpec.LONDON:
        backend = FakeLondon()
    elif backend_ == HardwareSpec.MANHATTAN:
        backend = FakeManhattan()
    elif backend_ == HardwareSpec.MANILA:
        backend = FakeManila()
    elif backend_ == HardwareSpec.MELBOURNE:
        backend = FakeMelbourne()
    elif backend_ == HardwareSpec.MONTREAL:
        backend = FakeMontreal()
    elif backend_ == HardwareSpec.OSLO:
        backend = FakeOslo()
    elif backend_ == HardwareSpec.OURENSE:
        backend = FakeOurense()
    elif backend_ == HardwareSpec.JAKARTA:
        backend = FakeJakarta()
    elif backend_ == HardwareSpec.PARIS:
        backend = FakeParis()
    elif backend_ == HardwareSpec.QUITO:
        backend = FakeQuito()
    elif backend_ == HardwareSpec.POUGHKEEPSIE:
        backend = FakePoughkeepsie()
    elif backend_ == HardwareSpec.ROCHESTER:
        backend = FakeRochester()
    elif backend_ == HardwareSpec.ROME:
        backend = FakeRome()
    elif backend_ == HardwareSpec.RUESCHLIKON:
        backend = FakeRueschlikon()
    elif backend_ == HardwareSpec.SANTIAGO:
        backend = FakeSantiago()
    elif backend_ == HardwareSpec.SINGAPORE:
        backend = FakeSingapore()
    elif backend_ == HardwareSpec.SYDNEY:
        backend = FakeSydney()
    elif backend_ == HardwareSpec.TOKYO:
        backend = FakeTokyo()
    elif backend_ == HardwareSpec.TORONTO:
        backend = FakeToronto()
    elif backend_ == HardwareSpec.VIGO:
        backend = FakeVigo()
    elif backend_ == HardwareSpec.WASHINGTON:
        backend = FakeWashington()
    elif backend_ == HardwareSpec.YORKTOWN:
        backend = FakeYorktown()
    elif backend_ == HardwareSpec.JAKARTA:
        backend = FakeJakarta()
    else:
        raise Exception("Could not retrieve backend", hardware_spec)
    ibm_noise_model = IBMNoiseModel.from_backend(backend, thermal_relaxation=thermal_relaxation)
    return ibm_noise_model

class Instruction:
    target: int
    control: int
    op: Op
    params: Any
    def __init__(self, target: int, op: Op, control: Optional[int] = None, params: Any = None) -> None:
        assert isinstance(op, Op)
        assert isinstance(target, int)
        assert isinstance(control, int) or (control is None)
        self.target = target
        self.op = op
        if not is_multiqubit_gate(op) and (control is not None):
            raise Exception("controls are initialized in multiqubit gate")
        elif op == Op.CNOT and control is None:
            raise Exception("CNOT gate should have exactly 1 control qubit")
        if target == control:
            raise Exception("target is in controls")
        self.control = control
        self.params = params

    def name(self, embedding):
        inverse_embedding = invert_dict(embedding)
        for (key, value) in embedding.items():
            assert value not in inverse_embedding.keys()
            inverse_embedding[value] = key

        if self.control is None:
            return f"{self.op.name}-{inverse_embedding[self.target]}"
        else:
            return f"{self.op.name}-{inverse_embedding[self.control]}-{inverse_embedding[self.target]}"
        
    def get_control(self, embedding)->str:
        inverse_embedding = invert_dict(embedding)
        if self.control is None:
            return ""
        else:
            return str(inverse_embedding[self.control])
        
    def get_target(self, embedding)->str:
        inverse_embedding = invert_dict(embedding)
        return str(inverse_embedding[self.target])
    
    def get_gate_data(self, is_meas_0=None):
        if self.is_meas_instruction():
            assert self.control is None
            assert is_meas_0 is not None
            if is_meas_0:
                return GateData(Op.P0, self.target)
            else:
                return GateData(Op.P1, self.target)
        else:
            assert is_meas_0 is None
        return GateData(self.op, self.target, self.control, self.params)
    
    
    def is_meas_instruction(self):
        return self.op in [Op.MEAS]

    def __eq__(self, value: object) -> bool:
        if isinstance(value, KrausOperator):
            return False
        return self.target == value.target and self.control == value.control and self.op == value.op and self.params == value.params
    
    def __hash__(self):
        return hash((self.op.value, self.target, self.control, self.params))
    
    def serialize(self):
        return {
            'type': 'instruction',
            'target': self.target,
            'control': self.control,
            'op': self.op.value,
            'params': self.params
        }
        
class KrausOperator:
    def __init__(self, operators, qubit) -> None:
        for operator in operators:
            assert operator.shape == (2,2) # for now we are dealing only with single qubit operators
        self.operators = operators # these are matrices
        self.target = qubit

    def serialize(self):
        serialized_operators = []
        for op in self.operators:
            curr_op = []
            for l in op:
                temp_l = []
                for element in l:
                    temp_l.append({'real': element.real, 'im': element.imag})
                curr_op.append(temp_l)
            serialized_operators.append(curr_op)
            

        return {
            'type': 'kraus',
            'target': self.target,
            'ops': serialized_operators,
        }
    
def is_identity(seq: List[Op]):
    for s in seq:
        assert(isinstance(s, Instruction))
        if s.op != Op.I:
            return False
    return True

class QuantumChannel:
    def __init__(self, all_ins_sequences, all_probabilities, target_qubits, optimize=False, flatten=True) -> None:
        self.errors = [] # list of list of sequences of instructions/kraus operators
        self.probabilities = all_probabilities
        for seq in all_ins_sequences:
            new_seq = QuantumChannel.translate_err_sequence(seq, target_qubits, optimize)
            self.errors.append(new_seq)
        assert len(self.errors) == len(self.probabilities)

        if optimize:
            self.errors, self.probabilities = QuantumChannel.remove_duplicates(self.errors, self.probabilities)

        if flatten:
            self.flatten()
        
        self.__check_probabilities()

        self.estimated_success_prob = self._get_success_probability()

    def __check_probabilities(self):
        for p in self.probabilities:
            assert 0.0 < p <= 1.0

    def _get_success_probability(self):
        temp = []
        temp_ins = []
        for (index, instruction) in enumerate(self.errors):
            if is_identity(instruction) or self.probabilities[index] > 0.5:
                temp_ins.append(instruction)
                if Precision.is_lowerbound:
                    temp.append(float(myfloor(self.probabilities[index], Precision.PRECISION)))
                else:
                    temp.append(float(myceil(self.probabilities[index], Precision.PRECISION)))
        if len(temp) == 0:
            temp.append(0.0)
        assert len(temp) == 1
        return temp[0]

    @staticmethod
    def flatten_sequence(err_seq):
        sequences = []
        for err in err_seq:
            if isinstance(err, Instruction):
                if len(sequences) == 0:
                    sequences.append([err])
                else:
                    for seq in sequences:
                        seq.append(err)
            else:
                assert isinstance(err, KrausOperator)
                if len(sequences) == 0:
                    for matrix in err.operators:
                        sequences.append([Instruction(err.target, Op.CUSTOM, params=matrix)])
                else:
                    all_seqs_temp = []
                    for seq in sequences:
                        for matrix in err.operators:
                            temp_seq = deepcopy(seq)
                            temp_seq.append(Instruction(err.target, Op.CUSTOM, params=matrix))
                            all_seqs_temp.append(temp_seq)

                    sequences = all_seqs_temp
                

        assert len(sequences) > 0
        return sequences

    def flatten(self):
        new_probabilities = []
        new_errors = []

        for (err_seq, prob) in zip(self.errors, self.probabilities):
            flattened_sequences = QuantumChannel.flatten_sequence(err_seq)

            for flattened_seq in flattened_sequences:
                new_probabilities.append(prob)
                new_errors.append(flattened_seq)

        self.errors = new_errors
        self.probabilities = new_probabilities


    def serialize(self):
        serialized_errors = []
        for err_seq in self.errors:
            temp_seq = []
            for e in err_seq:
                temp_seq.append(e.serialize())
            serialized_errors.append(temp_seq)
        return {
            'probabilities': self.probabilities,
            'errors': serialized_errors
        }
    
    @staticmethod
    def remove_duplicates(errors: List[List[Instruction]], probabilities: List[float]):
        """removes identical sequences of errors

        Args:
            errors (_type_): _description_
            probabilities (_type_): _description_

        Returns:
            _type_: _description_
        """
        new_errors = []
        def is_error_in_list(err):
            for (index,e) in enumerate(new_errors):
                if e == err:
                    return index
            return None

        new_probabilities = []
        for (err, p) in zip(errors, probabilities):
            assert len(new_probabilities) == len(new_errors)
            index = is_error_in_list(err)
            if index is not None:
                new_probabilities[index] += p
            else:
                new_errors.append(err)
                new_probabilities.append(p)
        return new_errors, new_probabilities

    @staticmethod
    def optimize_pauli_seq(seq):
        paulis_counts = dict()
        for instruction in seq:
            assert is_pauli(instruction.op)
            if instruction not in paulis_counts.keys():
                paulis_counts[instruction] = 0
            paulis_counts[instruction] += 1
        answer = []

        for (instruction, count) in paulis_counts.items():
            assert isinstance(instruction, Instruction)
            if count % 2 == 1:
                answer.append(instruction)
        
        return sorted(answer, key=lambda x: (x.target, x.op.value))

    @staticmethod
    def optimize_err_sequence(err_seq):
        # remove all identities
        new_seq1 = []
        for instruction in err_seq:
            if isinstance(instruction, KrausOperator) or instruction.op != Op.I:
                new_seq1.append(instruction)

        # replace Y gates for XZ (its the same up to a global phase)
        new_seq2 = []
        for instruction in new_seq1:
            if isinstance(instruction, KrausOperator) or instruction.op != Op.Y:
                new_seq2.append(instruction)
            else:
                assert instruction.op == Op.Y
                assert instruction.control is None
                new_seq2.append(Instruction(instruction.target, Op.X))
                new_seq2.append(Instruction(instruction.target, Op.Z))
                
        # optimize pauli sequences of gates: since pauli commute up to a global factor, we try to exploit that to have shorter error sequences
        temp_seq = []
        new_seq3 = []
        for instruction in new_seq2:
            if isinstance(instruction, KrausOperator) or instruction.op == Op.RESET:
                new_seq3.extend(QuantumChannel.optimize_pauli_seq(temp_seq))
                temp_seq = []
                new_seq3.append(instruction)
            else:
                assert is_pauli(instruction.op)
                temp_seq.append(instruction)
        new_seq3.extend(QuantumChannel.optimize_pauli_seq(temp_seq))
        return new_seq3
    
    @staticmethod
    def translate_err_sequence(err_seq, target_qubits, optimize):
        answer = []
        for err in err_seq:
            if err['name'] == 'pauli':
                assert len(target_qubits) == 2
                assert len(err['params']) == 1
                assert len(err['params'][0]) == 2 # II, IX, IZ, XX, etc
                for (p, qubit) in zip(err['params'][0], err['qubits']):
                    op = get_op(p)
                    target_qubit = target_qubits[qubit]
                    answer.append(Instruction(target_qubit, op))
            elif err['name'] == 'kraus':
                assert len(err['qubits']) == 1
                answer.append(KrausOperator(err['params'], target_qubits[err['qubits'][0]]))
            else:
                op = get_op(err['name'])
                assert len(err['qubits']) == 1
                target_qubit = target_qubits[err['qubits'][0]]
                answer.append(Instruction(target_qubit, op))
        if optimize:
            return QuantumChannel.optimize_err_sequence(answer)
        else:
            return answer


class MeasChannel:
    def __init__(self, all_probabilities) -> None:
        assert len(all_probabilities) == 2
        self.meas_errors = dict()

        zero_meas_err = all_probabilities[0]
        assert len(zero_meas_err) == 2
        self.meas_errors[0] = dict()
        self.meas_errors[0][0] = zero_meas_err[0] # probability that measurement outcome is 0 given that the ideal outcome should have been 0
        self.meas_errors[0][1] = zero_meas_err[1] # probability that measurement outcome is 0 given that the ideal outcome should have been 1

        one_meas_err = all_probabilities[1]
        assert len(one_meas_err) == 2
        self.meas_errors[1] = dict()
        self.meas_errors[1][0] = one_meas_err[0] # probability that measurement outcome is 1 given that the ideal outcome should have been 0
        self.meas_errors[1][1] = one_meas_err[1] # probability that measurement outcome is 0 given that the ideal outcome should have been 1
    
    def get_ind_probability(self, ideal_outcome: int, noisy_outcome: int):
        assert ideal_outcome in [0, 1]
        assert noisy_outcome in [0, 1]
        return self.meas_errors[noisy_outcome][ideal_outcome]
    
    def serialize(self):
        return self.meas_errors
            

class NoiseModel:
    hardware_spec: HardwareSpec
    basis_gates: List[Op]
    instructions_to_channel: Dict[Instruction, QuantumChannel|MeasChannel]
    num_qubits: int
    qubit_to_indegree: List[int, int] # tells mutiqubit gates have as target a given qubit (key)
    def __init__(self, hardware_specification: HardwareSpec, thermal_relaxation=True) -> None:
        self.hardware_spec = hardware_specification
        ibm_noise_model = get_ibm_noise_model(hardware_specification, thermal_relaxation=thermal_relaxation)
        self.basis_gates = get_basis_gate_type([get_op(op) for op in ibm_noise_model.basis_gates])
        self.instructions_to_channel = dict()
        try:
            self.num_qubits = ibm_noise_model.configuration().num_qubits
        except:
            self.num_qubits = ibm_noise_model.num_qubits

        self.qubit_to_indegree = dict()
        # start translating quantum channels
        all_errors = ibm_noise_model.to_dict()
        assert len(all_errors.keys()) == 1

        all_errors = all_errors['errors']

        for error in all_errors:
            target_instructions = error['operations'] # this error applies to these instructions
            assert len(target_instructions) == 1 # we are assumming that errors target only 1       instruction at once
            op = get_op(target_instructions[0])

            assert len(error['gate_qubits']) == 1
            error_target_qubits = error['gate_qubits'][0] # this error targets the following qubits
            control = None
            if len(error_target_qubits) > 1:
                assert len(error_target_qubits) == 2 # the only gates for multiqubit gates at IBM are CX gates, therefore at most, this error targets 2 qubits
                control = error_target_qubits[0]
                target = error_target_qubits[1]
                target_qubits = [control, target]

                assert op in [Op.CNOT, Op.CZ]
                if target not in self.qubit_to_indegree.keys():
                    self.qubit_to_indegree[target] = 0
                self.qubit_to_indegree[target] += 1
            else:
                target = error_target_qubits[0]
                target_qubits = [target]
                
            target_instruction = Instruction(target, op, control)
            probabilities = error['probabilities']
            if error['type'] == "qerror":    
                error_instructions = error['instructions']
                self.instructions_to_channel[target_instruction] = QuantumChannel(error_instructions, probabilities, target_qubits)
            else:
                assert error['type'] == "roerror"
                self.instructions_to_channel[target_instruction] = MeasChannel(probabilities)

        assert len(self.qubit_to_indegree.keys()) == self.num_qubits
    
    def get_qubit_couplers(self, target: int) -> List[int]:
        ''' Returns a list of pairs (instruciton, QuantumChannel) in which the instruction is a multiqubit gate whose target is the given qubit
        '''
        assert (target >= 0)
        result = []

        for (instruction, channel) in self.instructions_to_channel.items():
            assert isinstance(instruction, Instruction)
            if is_multiqubit_gate(instruction.op):
                assert isinstance(instruction.target, int)
                assert isinstance(instruction.control, int)
                if target == instruction:
                    result.append((instruction, channel))

        result = sorted(result, key=lambda x : x[1].estimated_success_prob, reverse=False)
        return result
    
    def serialize(self):
        instructions = []
        channels = []
        for (instruction, channel) in self.instructions_to_channel.items():
            instructions.append(instruction.serialize())
            channels.append(channel.serialize())

        assert len(instructions) == len(channels)
        return {
            'hardware': self.hardware_spec.value,
            'basis_gates': [x.value for x in self.basis_gates],
            'instructions': instructions,
            'channels': channels
        }
    
    def dump_json(self, path):
        f = open(path, "w")
        f.write(json.dumps(self.serialize(), indent=4))
        f.close()

if __name__ == "__main__":
    pass
