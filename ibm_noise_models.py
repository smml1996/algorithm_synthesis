from cmath import cos, sin, sqrt
from copy import deepcopy
from enum import Enum
from math import pi
from typing import Any, Dict, List, Optional
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import *
from qiskit_aer.noise import NoiseModel as IBMNoiseModel
from qiskit_aer import AerSimulator
from qiskit.extensions import XGate, ZGate, CXGate, UGate, SXGate, RZGate
from qpu_utils import *
import json

from utils import CONFIG_KEYS, Precision, find_enum_object, invert_dict, myceil, myfloor


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
    # PRAGUE = "fake_prague"
    ALMADEN = "fake_almaden"
    # ARMONK = "fake_armonk"
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
    # RUESCHLIKON = "fake_rueschlikon"
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

def load_config_file(path: str, experimentID: Enum):
    f = open(path)
    result = json.load(f)
    for k in CONFIG_KEYS:
        if k not in result.keys():
            raise Exception(f"{k} not in config file")
    assert isinstance(result["min_horizon"], int)
    assert isinstance(result["max_horizon"], int)
    
    if result["output_dir"][-1] != "/" :
        result["output_dir"] += "/"
     
    experiment_id = find_enum_object(result["experiment_id"], experimentID)
    if experiment_id is None:
        raise Exception(f"Invalid experiment_id (available values {[x.value for x in experimentID]})")
    
    if result["hardware"] == ["all"] or result["hardware"] == "all":
        result["hardware"] = [x.value for x in HardwareSpec]
    result["experiment_id"] = experiment_id
    f.close()
    return result

def get_ibm_noise_model(hardware_spec: HardwareSpec, thermal_relaxation=True) -> IBMNoiseModel:
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
    # elif backend_ == HardwareSpec.PRAGUE:
    #     backend = FakePrague()
    elif backend_ == HardwareSpec.ALMADEN:
        backend = FakeAlmaden()
    # elif backend_ == HardwareSpec.ARMONK:
    #     backend = FakeArmonk()
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
    # elif backend_ == HardwareSpec.RUESCHLIKON:
    #     backend = FakeRueschlikon()
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
    def __init__(self, target: int, op: Op, control: Optional[int] = None, params: Any = None, name=None, symbols=None) -> None:
        assert isinstance(op, Op)
        assert isinstance(target, int)
        assert isinstance(control, int) or (control is None)
        self.target = target
        self.op = op
        if (not is_multiqubit_gate(op)) and (control is not None):
            raise Exception(f"controls are initialized in a non-multiqubit gate ({op} {control})")
        elif is_multiqubit_gate(op) and control is None:
            raise Exception(f"{op} gate should have exactly 1 control ({control}) qubit")
        if target == control:
            raise Exception("target is in controls")
        self.control = control
        if params is not None:
            for pa in params:
                assert isinstance(pa, float) or isinstance(pa, str)
        self.params = params
        self.symbols = symbols
        if symbols is not None:
            temp_set = set(symbols)
            if len(self.symbols) != len(temp_set):
                assert len(temp_set) < len(self.symbols)
                raise Exception(f"There are repeated symbols: {symbols}")
        self.name = name
        
    def is_identity(self):
        for p in self.params:
            assert not isinstance(p, complex)
            assert not isinstance(p, str)
            
        matrix_values = []
        if self.Op.U1:
            matrix_values.append(1)
            matrix_values.append(0)
            matrix_values.append(0)
            matrix_values.append(np.e**(complex(0, self.params[0])))
        if self.Op.U1D:
            matrix_values.append(1)
            matrix_values.append(0)
            matrix_values.append(0)
            matrix_values.append(np.e**(complex(0, -self.params[0])))
        if self.op in [Op.U2, Op.U2D]:
            return False
        if self.op == Op.U3:
            matrix_values.append(cos(self.params[0]/2))
            matrix_values.append(sin(self.params[0]/2))
            matrix_values.append(sin(self.params[0]/2))
            matrix_values.append(np.e**(complex(0,self.params[1] + self.params[2]))*cos(self.params[0]/2))
        if self.op == Op.U3D:
            matrix_values.append(cos(self.params[0]/2))
            matrix_values.append(sin(self.params[0]/2))
            matrix_values.append(sin(self.params[0]/2))
            matrix_values.append(np.e**(complex(0,-self.params[1] - self.params[2]))*cos(self.params[0]/2))
        if self.Op in [Op.RX, Op.RY]:
            matrix_values.append(cos(self.params[0]/2))
            matrix_values.append(sin(self.params[0]/2))
            matrix_values.append(sin(self.params[0]/2))
            matrix_values.append(cos(self.params[0]/2))
        if self.Op.RZ:
            matrix_values.append(np.e**(complex(0, -self.params[0]/2)))
            matrix_values.append(0)
            matrix_values.append(0)
            matrix_values.append(np.e**(complex(0, self.params[0]/2)))
            
        for index in range(len(matrix_values)):            
            matrix_values[index] = matrix_values[index] * np.conjugate(matrix_values[index])
            assert isclose(matrix_values[index].imag, 0, abs_tol=Precision.isclose_abstol)
            matrix_values[index] = matrix_values[index].real

        if isclose(matrix_values[0], matrix_values[3], rel_tol=Precision.rel_tol):
            if not isclose(abs(matrix_values[0]), 1, rel_tol=Precision.rel_tol):
                return False
        else:
            return False
        if not isclose(matrix_values[1], 0, abs_tol=Precision.isclose_abstol):
            return False
        if not isclose(matrix_values[2], 0, abs_tol=Precision.isclose_abstol):
            return False

        return False
        
        
    def bind_symbols_from_lst(self, values: List[float]) -> Any:
        """_summary_

        Args:
            values (List[float]): _description_

        Returns:
            Any: returns an instruction with the parameters binded
        """
        
        assert len(values) == len(self.symbols)
        d = zip(self.symbols, values)
        
        return self.bind_symbols_from_dict(d)
    
    def bind_symbols_from_dict(self, d: Dict[str, float]) -> Any:
        if self.params is None:
            return self
        
        new_params = []
        
        for p in self.params:
            if isinstance(p, str):
                new_params.append(d[p])
            else:
                new_params.append(p)
        return Instruction(self.target, self.op, self.control, params=new_params)
    
    def is_classical(self):
        return self.op in [Op.WRITE0, Op.WRITE1, Op.TOGGLE]

    def name(self, embedding):
        if self.name is not None:
            return self.name
        inverse_embedding = invert_dict(embedding)
        for (key, value) in embedding.items():
            assert value not in inverse_embedding.keys()
            inverse_embedding[value] = key

        if self.control is None:
            return f"{self.op.name}-{inverse_embedding[self.target]}"
        else:
            return f"{self.op.name}-{inverse_embedding[self.control]}-{inverse_embedding[self.target]}"
    
    def get_num_parameters(self) -> int:
        return len(self.symbols)
        
    def get_control(self, embedding)->str:
        inverse_embedding = invert_dict(embedding)
        if self.control is None:
            return ""
        else:
            return str(inverse_embedding[self.control])
        
    def get_target(self, embedding)->str:
        inverse_embedding = invert_dict(embedding)
        return str(inverse_embedding[self.target])
    
    def get_params(self) -> str:
        if self.params is None:
            return "-"
        if isinstance(self.params, list):
            return ";".join([str(x) for x in self.params])
        return self.params
    
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
        return self.target == value.target and self.control == value.control and self.op == value.op
    
    def __hash__(self):
        return hash((self.op.value, self.target, self.control))
    
    def serialize(self, for_json=False):
        if for_json:
            return self.name
        else:
            return {
                'type': 'instruction',
                'target': self.target,
                'control': self.control,
                'op': self.op,
                'params': self.params
            }
        
    def __str__(self) -> str:
        return f"Instruction(target={self.target}, control={self.control}, op={self.op}, params={self.params})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def to_basis_gate_impl(self, basis_gates: BasisGates) -> List[Any]:
        """_summary_

        Args:
            basis_gates (BasisGates): _description_

        Raises:
            Exception: _description_

        Returns:
            List[Instruction]:  returns a list of instructions corresponding to the native gate sequence implementation using basis_gates
        """        
        if (self.op in basis_gates.value) or (self.op == Op.MEAS):
            return [self]
        
        if basis_gates in [BasisGates.TYPE1, BasisGates.TYPE6]:
            if self.op == Op.H:
                return [Instruction(self.target, Op.U2, params=[0.0, pi])]
            if self.op == Op.T:
                return [Instruction(self.target, Op.U1, params=[pi/4])]
            if self.op == Op.S:
                return [Instruction(self.target, Op.U1, params=[pi/2])]
            if self.op == Op.RX:
                return [Instruction(self.target, Op.U3, params=[self.params[0], -pi/2, pi/2], symbols=self.symbols)]
            if self.op == Op.RY:
                return [Instruction(self.target, Op.U3, params=[self.params[0], 0.0, 0.0], symbols=self.symbols)]
            if self.op == Op.RZ:
                return [Instruction(self.target, Op.U1, params=[self.params[0]], symbols=self.symbols)]
            if self.op == Op.X:
                assert basis_gates != BasisGates.TYPE6
                return [Instruction(self.target, Op.U3, params=[pi, 0.0, pi], symbols=self.symbols)]
        else:
            assert basis_gates in [BasisGates.TYPE2, BasisGates.TYPE3, BasisGates.TYPE7]
            if self.op == Op.H:
                return [Instruction(self.target, Op.RZ, params=[pi/2]),
                Instruction(self.target, Op.SX),
                Instruction(self.target, Op.RZ, params=[pi/2])]
            if self.op == Op.U3:
                print("params", self.params)
                ry_symbols = None
                rz_symbols2 = None
                rz_symbols1 = None

                if isinstance(self.params[0], str):
                    ry_symbols = [self.params[0]]
                    assert self.params[0] in self.symbols
                if isinstance(self.params[1], str):
                    rz_symbols1 = [self.params[1]]
                if isinstance(self.params[2], str):
                    rz_symbols2 = [self.params[2]]
                    
                rz_lambda = [Instruction(self.target, Op.RZ, params=[self.params[2]], symbols=rz_symbols2)]
                ry_theta = Instruction(self.target, Op.RY, params=[self.params[0]], symbols=ry_symbols).to_basis_gate_impl(basis_gates)
                rz_phi = [Instruction(self.target, Op.RZ, params=[self.params[1]], symbols=rz_symbols1)]
                return rz_lambda + ry_theta + rz_phi
            if self.op == Op.T:
                return [Instruction(self.target, Op.RZ, params=[pi/4])]
            if self.op == Op.S:
                return [Instruction(self.target, Op.RZ, params=[-pi/2])]
            if self.op == Op.SD:
                return [Instruction(self.target, Op.RZ, params=[pi/2])]
            else:
                h_gate = Instruction(self.target, Op.H).to_basis_gate_impl(basis_gates)
                rz = Instruction(self.target, Op.RZ, params=self.params, symbols=self.symbols).to_basis_gate_impl(basis_gates)
                if self.op == Op.RX:
                    return h_gate + rz + h_gate
                if self.op == Op.RY:
                    s_gate = Instruction(self.target, Op.S).to_basis_gate_impl(basis_gates)
                    s_inverse = Instruction(self.target, Op.SD).to_basis_gate_impl(basis_gates)
                    answer = s_gate + h_gate + rz + h_gate + s_inverse
                    return answer
                
        raise Exception(f"Cannot translate {self.op} to basis gates {basis_gates}")
        
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
    
def is_identity(seq: List[GateData]):
    for s in seq:
        assert(isinstance(s, GateData))
        assert isinstance(s.label, Op)
        if s.label != Op.I:
            return False
    return True

class QuantumChannel:
    def __init__(self, all_ins_sequences, all_probabilities, target_qubits, optimize=False, flatten=False) -> None:
        self.errors = [] # list of list of sequences of instructions/kraus operators
        self.probabilities = all_probabilities
        for seq in all_ins_sequences:
            new_seq = QuantumChannel.translate_err_sequence(seq, target_qubits, optimize)
            self.errors.append(new_seq)
        assert len(self.errors) == len(self.probabilities)

        if optimize:
            assert False # TODO: Remove me
            self.errors, self.probabilities = QuantumChannel.remove_duplicates(self.errors, self.probabilities)

        if len(self.probabilities) == 0:
            self.probabilities = [1.0]
            self.errors = [[Instruction(target_qubits[0], Op.I).get_gate_data()]]
        else:
            assert len(self.errors) > 0

        if flatten:
            assert False # TODO: Remove me
            self.flatten()

        self.estimated_success_prob = self._get_success_probability()
        
        self.__check_probabilities()
        
    def __str__(self) -> str:
        return {"type": "QuantumChannel", "errors": self.errors, "probs":self.probabilities}.__str__()
    
    def __repr__(self):
        return self.__str__()

    def __check_probabilities(self):
        assert len(self.probabilities) > 0
        for p in self.probabilities:
            assert 0.0 < p <= 1.0

    def _get_success_probability(self):
        temp = 0.5
        for (index, instruction) in enumerate(self.errors):
            if is_identity(instruction) or self.probabilities[index] > 0.5:
                if Precision.is_lowerbound:
                    temp = max(temp, float(myfloor(self.probabilities[index], Precision.PRECISION)))
                else:
                    temp = max(temp, float(myceil(self.probabilities[index], Precision.PRECISION)))
        return temp

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
                assert False
                assert isinstance(err, KrausOperator)
                if len(sequences) == 0:
                    for matrix in err.operators:
                        sequences.append([Instruction(err.target, Op.CUSTOM, params=matrix).get_gate_data()])
                else:
                    all_seqs_temp = []
                    for seq in sequences:
                        for matrix in err.operators:
                            temp_seq = deepcopy(seq)
                            temp_seq.append(Instruction(err.target, Op.CUSTOM, params=matrix).get_gate_data())
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
        assert False # TODO: remove this
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
        assert not optimize # TODO: remove me
        answer = []
        for err in err_seq:
            if err['name'] == 'pauli':
                assert len(target_qubits) == 2
                assert len(err['params']) == 1
                assert len(err['params'][0]) == 2 # II, IX, IZ, XX, etc
                for (p, qubit) in zip(err['params'][0], err['qubits']):
                    op = get_op(p)
                    target_qubit = target_qubits[qubit]
                    answer.append(Instruction(target_qubit, op).get_gate_data())
            elif err['name'] == 'kraus':
                assert len(err['qubits']) == 1
                answer.append(KrausOperator(err['params'], target_qubits[err['qubits'][0]]))
            else:
                op = get_op(err['name'])
                assert len(err['qubits']) == 1
                target_qubit = target_qubits[err['qubits'][0]]
                answer.append(Instruction(target_qubit, op).get_gate_data())
        if optimize:
            assert False # TODO: remove this
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
        self.meas_errors[0][1] = zero_meas_err[1] # probability that measurement outcome is 1 given that the ideal outcome should have been 0

        one_meas_err = all_probabilities[1]
        assert len(one_meas_err) == 2
        self.meas_errors[1] = dict()
        self.meas_errors[1][0] = one_meas_err[0] # probability that measurement outcome is 0 given that the ideal outcome should have been 1
        self.meas_errors[1][1] = one_meas_err[1] # probability that measurement outcome is 1 given that the ideal outcome should have been 1
    
    def get_success_probability(self):
        return self.get_ind_probability(0,0) + self.get_ind_probability(1,1)
        
    def get_ind_probability(self, ideal_outcome: int, noisy_outcome: int):
        assert ideal_outcome in [0, 1]
        assert noisy_outcome in [0, 1]
        return self.meas_errors[ideal_outcome][noisy_outcome]
    
    def serialize(self):
        return self.meas_errors
    
    def __str__(self) -> str:
        return {"type": "MeasChannel", "errors": self.meas_errors}.__str__()
    
    def __repr__(self):
        return self.__str__()
            

class NoiseModel:
    hardware_spec: HardwareSpec
    basis_gates: List[Op]
    instructions_to_channel: Dict[Instruction, QuantumChannel|MeasChannel]
    num_qubits: int
    qubit_to_indegree: Dict[int, int] # tells mutiqubit gates have as target a given qubit (key)
    qubit_to_outdegree: Dict[int, int]
    def __init__(self, hardware_specification: HardwareSpec, thermal_relaxation=True) -> None:
        self.hardware_spec = hardware_specification
        ibm_noise_model = get_ibm_noise_model(hardware_specification, thermal_relaxation=thermal_relaxation)
        assert isinstance(ibm_noise_model, IBMNoiseModel)
        self.basis_gates = get_basis_gate_type([get_op(op) for op in ibm_noise_model.basis_gates])
        self.instructions_to_channel = dict()
        self.num_qubits = len(ibm_noise_model.noise_qubits)

        self.qubit_to_indegree = dict()
        self.qubit_to_outdegree = dict()
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
                if control not in self.qubit_to_outdegree.keys():
                    self.qubit_to_outdegree[control] = 0
                self.qubit_to_indegree[target] += 1
                self.qubit_to_outdegree[control] += 1
            else:
                target = error_target_qubits[0]
                target_qubits = [target]
                
            target_instruction = Instruction(target, op, control)
            probabilities = error['probabilities']
            if error['type'] == "qerror":    
                error_instructions = error['instructions']
                self.instructions_to_channel[target_instruction] = QuantumChannel(error_instructions, probabilities, target_qubits)
                
                # TODO: remove this
                # if op == Op.CNOT and target_instruction.control==1 and target_instruction.target== 0:
                #     print(hardware_specification, target_instruction)
                #     print(probabilities, "\n")
                
            else:
                assert error['type'] == "roerror"
                self.instructions_to_channel[target_instruction] = MeasChannel(probabilities)
        
        # check that all single qubit gates exist
        report = dict()
        for qubit in range(self.num_qubits):
            for op in self.basis_gates.value:
                assert isinstance(op, Op)
                if not is_multiqubit_gate(op):
                    instruction_ = Instruction(qubit, op)
                    if instruction_ not in self.instructions_to_channel.keys():
                        if op not in report.keys():
                            report[op] = 0
                        report[op] += 1

                        # create a perfect quantum channel for this operation
                        self.instructions_to_channel[instruction_] = QuantumChannel([], [], [qubit])
        self.report = report
        self.digraph = self.get_digraph_()
        # if len(report.keys()) > 0:
        #     print(f"WARNING ({hardware_specification.value}) (qubits={self.num_qubits}) ({self.basis_gates.value}): no quantum channel found for {report}")

    def get_digraph_(self):
        answer = dict()
        for instruction in self.instructions_to_channel.keys():
            if is_multiqubit_gate(instruction.op):
                source = instruction.control
                target = instruction.target
                
                if source not in answer.keys():
                    answer[source] = set()
                answer[source].add(target)
        return answer
            
    def get_qubit_indegree(self, qubit) -> int:
        if qubit in self.qubit_to_indegree.keys():
            return self.qubit_to_indegree[qubit]
        else:
            return 0
        
    def get_qubit_outdegree(self, qubit) -> int:
        if qubit in self.qubit_to_outdegree.keys():
            return self.qubit_to_outdegree[qubit]
        else:
            return 0
        
    def get_qubit_couplers(self, target: int, is_target=True) -> List[int]:
        ''' Returns a list of pairs (qubit_control, QuantumChannel) in which the instruction is a multiqubit gate whose target is the given qubit
        '''
        assert (target >= 0)
        result = []

        for (instruction, channel) in self.instructions_to_channel.items():
            assert isinstance(instruction, Instruction)
            if is_multiqubit_gate(instruction.op):
                assert isinstance(instruction.target, int)
                assert isinstance(instruction.control, int)
                if is_target:
                    if target == instruction.target:
                        result.append((instruction.control, channel))
                else:
                    if target == instruction.control:
                        result.append((instruction.target, channel))

        result = sorted(result, key=lambda x : x[1].estimated_success_prob, reverse=False)
        return result
    
    def get_most_noisy_couplers(self):
        result = []
        for (instruction, channel) in self.instructions_to_channel.items():
            assert isinstance(instruction, Instruction)
            if is_multiqubit_gate(instruction.op):
                assert isinstance(instruction.target, int)
                assert isinstance(instruction.control, int)
                result.append(((instruction.control, instruction.target), channel))

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
        
    # functions that help to choose embeddings follow
    def get_most_noisy_qubit(self, op: Op, top=1, reverse=False) -> List[int]:
        
        assert (op in self.basis_gates.value) or (op == Op.MEAS)
        
        qubits_and_noises = []
        for (instruction, channel) in self.instructions_to_channel.items():
            if instruction.op == op:
                if isinstance(channel, QuantumChannel):
                    if is_multiqubit_gate(op):
                        qubits_and_noises.append((channel.estimated_success_prob, (instruction.target, instruction.control)))
                    else:
                        qubits_and_noises.append((channel.estimated_success_prob, instruction.target))
                else:
                    assert isinstance(channel, MeasChannel)
                    qubits_and_noises.append((channel.get_success_probability()/2.0, instruction.target))
                    
        qubits_and_noises = sorted(qubits_and_noises, key=lambda x : x[0], reverse=reverse)
        return qubits_and_noises
            
            
NoiselessX = XGate(label="noiseless_x")
NoiselessZ = ZGate(label="noiseless_z")
NoiselessCX = CXGate(label="noiseless_cx")
NoiselessSX = SXGate(label="noiseless_sx")          

def instruction_to_ibm(qc, instruction_sequence, noiseless=False):
    for instruction in instruction_sequence:
        assert isinstance(instruction, Instruction)
        assert isinstance(qc, QuantumCircuit)
        if instruction.op == Op.X:
            if noiseless:
                qc.append(NoiselessX, [instruction.target])
            else:
                qc.x(instruction.target)
        elif instruction.op == Op.Z:
            if noiseless:
                qc.append(NoiselessZ, [instruction.target])
            else:
                qc.z(instruction.target)
        elif instruction.op == Op.MEAS:
            assert not noiseless
            qc.measure(instruction.target, instruction.target)
        elif instruction.op == Op.CNOT:
            assert instruction.control is not None
            if noiseless:
                qc.append(NoiselessCX, [instruction.control, instruction.target])
            else:
                qc.cx(instruction.control, instruction.target)
        elif instruction.op == Op.U3:
            if noiseless:
                NoiselessU3 = UGate(label="noiseless_u3", theta=instruction.params[0], phi=instruction.params[1], lam=instruction.params[2])
                qc.append(NoiselessU3, [instruction.target])
            else:
                qc.u(instruction.params[0], instruction.params[1], instruction.params[2], instruction.target)
        elif instruction.op == Op.U2:
            if noiseless:
                NoiselessU3 = UGate(label="noiseless_u3", theta=pi/2, phi=instruction.params[0], lam=instruction.params[1])
                qc.append(NoiselessU3, [instruction.target])
            else:
                qc.u(pi/2, instruction.params[0], instruction.params[1], instruction.target)
        elif instruction.op == Op.SX:
            if noiseless:
                qc.append(NoiselessSX, [instruction.target])
            else:
                qc.sx(instruction.target)
        elif instruction.op == Op.RZ:
            if noiseless:
                NoiselessRZ = RZGate(label="noiseless_rz", phi=instruction.params[0])
                qc.append(NoiselessRZ, [instruction.params[0], instruction.target])
            else:
                qc.rz(instruction.params[0], instruction.target)
        else:
            if not instruction.is_classical():
                raise Exception(f"Instruction {instruction.name} could not be translated to IBM instruction. Missing implementation.")
    
def ibm_simulate_circuit(qc: QuantumCircuit, noise_model, initial_layout, seed=1):
    # Create noisy simulator backend
    sim_noise = AerSimulator(method ='statevector', noise_model=noise_model)
    # Transpile circuit for noisy basis gates
    circ_tnoise = transpile(qc, sim_noise, optimization_level=0, initial_layout=initial_layout)
    # Run and get counts
    
    result = sim_noise.run(circ_tnoise, run_options={"seed_simulator": seed}).result()
    
    return np.asarray(result.data()['res'])

def get_num_qubits_to_hardware(with_thermalization: bool, hardware_str=True, allowed_hardware=HardwareSpec) -> Dict[int, HardwareSpec|str]:
    s = dict()
    for hardware in allowed_hardware:
        nm = NoiseModel(hardware, thermal_relaxation=with_thermalization)
        if nm.num_qubits not in s.keys():
            s[nm.num_qubits] = []
        if hardware_str:
            s[nm.num_qubits].append(hardware.value) 
        else:
            s[nm.num_qubits].append(hardware) 
    return s


if __name__ == "__main__":
    pass
