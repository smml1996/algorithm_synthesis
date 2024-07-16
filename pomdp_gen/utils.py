from cmath import isclose
from collections import deque
from enum import Enum
import enum
from math import floor, ceil
from typing import Any, Dict, List, Optional
import random
from sympy import *
import numpy as np
from typing import Dict
from qiskit import QuantumCircuit, ClassicalRegister
import importlib
from importlib.machinery import SourceFileLoader

class Precision:
    PRECISION = 2  # round number to `PRECISION` floating point digits
    isclose_abstol = None
    rel_tol = None
    is_lowerbound = True
    @staticmethod
    def update_threshold():
        Precision.isclose_abstol = 0.00000000001
        Precision.rel_tol = 1/(10**(Precision.PRECISION-1))  


def get_inverse_mapping(m):
    result = {}
    for (key, val) in m.items():
        assert val not in result.keys()
        result[val] = key
    return result

# some IBM_backend_names
FAKE_TENERIFE = "fake_tenerife"
FAKE_JOHANNESBURG = "fake_johannesburg"
FAKE_PERTH = "fake_perth"
FAKE_LAGOS = "fake_lagos"
FAKE_NAIROBI = "fake_nairobi"
FAKE_HANOI = "fake_hanoi"
FAKE_CAIRO = "fake_cairo"
FAKE_MUMBAI = "fake_mumbai"
FAKE_KOLKATA = "fake_kolkata"
FAKE_PRAGUE = "fake_prague"
FAKE_SHERBROOKE = "fake_sherbrooke"
FAKE_ALMADEN = "fake_almaden"
FAKE_ARMONK = "fake_armonk"
FAKE_ATHENS = "fake_athens"
FAKE_AUCKLAND = "fake_auckland"
FAKE_BELEM = "fake_belem"
FAKE_BOEBLINGEN = "fake_boeblingen"
FAKE_BOGOTA = "fake_bogota"
FAKE_BROOKLYN = "fake_brooklyn"
FAKE_BURLINGTON = "fake_burlington"
FAKE_CAMBRIDGE = "fake_cambridge"
FAKE_CASABLANCA = "fake_casablanca"
FAKE_ESSEX = "fake_essex"
FAKE_GENEVA = "fake_geneva"
FAKE_GUADALUPE = "fake_guadalupe"
FAKE_LIMA = "fake_lima"
FAKE_LONDON = "fake_london"
FAKE_MANHATTAN = "fake_manhattan"
FAKE_MANILA = "fake_manila"
FAKE_MELBOURNE = "fake_melbourne"
FAKE_MONTREAL = "fake_montreal"
FAKE_OSLO = "fake_oslo"
FAKE_OURENSE = "fake_ourense"
FAKE_PARIS = "fake_paris"
FAKE_QUITO = "fake_quito"
FAKE_POUGHKEEPSIE = "fake_poughkeepsie"
FAKE_ROCHESTER = "fake_rochester"
FAKE_ROME = "fake_rome"
FAKE_RUESCHLIKON = "fake_rueschlikon"
FAKE_SANTIAGO = "fake_santiago"
FAKE_SINGAPORE = "fake_singapore"
FAKE_SYDNEY = "fake_sydney"
FAKE_TOKYO = "fake_tokyo"
FAKE_TORONTO = "fake_toronto"
FAKE_VIGO = "fake_vigo"
FAKE_WASHINGTON = "fake_washington"
FAKE_YORKTOWN = "fake_yorktown"
FAKE_JAKARTA = "fake_jakarta"
FAKE_CUSCO = "fake_cusco"

all_backends = [FAKE_TENERIFE, FAKE_JOHANNESBURG, FAKE_PERTH, FAKE_LAGOS, FAKE_NAIROBI, FAKE_HANOI, FAKE_CAIRO, FAKE_MUMBAI, FAKE_KOLKATA, FAKE_ALMADEN, FAKE_ATHENS, FAKE_AUCKLAND, FAKE_BELEM, FAKE_BOEBLINGEN, FAKE_BOGOTA, FAKE_BROOKLYN, FAKE_BURLINGTON, FAKE_CAMBRIDGE, FAKE_CASABLANCA, FAKE_ESSEX, FAKE_GENEVA, FAKE_GUADALUPE, FAKE_LIMA, FAKE_LONDON, FAKE_MANHATTAN, FAKE_MANILA, FAKE_MELBOURNE, FAKE_MONTREAL, FAKE_OSLO, FAKE_OURENSE, FAKE_PARIS, FAKE_QUITO, FAKE_POUGHKEEPSIE, FAKE_ROCHESTER, FAKE_ROME,  FAKE_SANTIAGO, FAKE_SINGAPORE, FAKE_SYDNEY, FAKE_TOKYO, FAKE_TORONTO, FAKE_VIGO, FAKE_WASHINGTON, FAKE_YORKTOWN, FAKE_JAKARTA]
# FAKE_PRAGUE, FAKE_SHERBROOKE, FAKE_ARMONK --> do not contain CX
# FAKE_RUESCHLIKON --> does not contain readout error model



DEBUG_BACKEND = "debug_backend"


def get_position(v, element) -> Optional[int]:
    for (index, e) in enumerate(v):
        if e == element:
            return index
    return None

def squared_norm(c):
    conj = conjugate(c)
    return simplify(c * conj)

def myfloor(val, d):
    m = 10**d
    return floor(val * m)/m

def myceil(val, d):
    m = 10**d
    return ceil(val * m)/m

def norm(c):
    sn = squared_norm(c)
    return simplify(sqrt(sn))


class Gate(Enum):
    # PAULI GATES
    X = "X"
    Y = "Y"
    Z = "Z"
    I = "I"

    # PHASE GATES
    T = "T"
    TD = "TD"  # The inverse of T
    S = "S"
    SD = "SD"
    SX = "SX"
    SXD = "SXD"

    U3 = "U3"
    U2 = "U2"
    U1 = "U1"
    # OTHER 1-QUBIT GATES
    XZ = "XZ"  # Z gate followed by an X Gate

    # HADAMARD
    H = "H"

    # MULTI-QUBIT GATES
    CNOT = "CNOT"
    CU3 = "CU3"
    CZ = "CZ"

    # PROJECTIONS
    P0 = "P0"  # projects a qubit to 1
    P1 = "P1"  # projects a qubit to 0

    # NON-UNITARY
    RESET = "RESET"

    # ClassicalOp
    WRITE0 = "WRITE0"
    WRITE1 = "WRITE1"

    def __repr__(self) -> str:
        return self.__str__()


def is_classical_op(gate: Gate):
    return gate in [Gate.WRITE0, Gate.WRITE1]


def is_projection(gate: Gate) -> bool:
    return gate in [Gate.P0, Gate.P1]


def have_same_keys(s1, s2) -> bool:
    for key in s1.sparse_vector.keys():
        if not (key in s2.sparse_vector):
            return False

    for key in s2.sparse_vector.keys():
        if not (key in s2.sparse_vector):
            return False
    return True


class GateData:
    label: Gate
    address: int
    controls: Optional[List[int]]
    params: Optional[List[float]]

    def __init__(self, label, address, controls=None, params=None) -> None:
        self.label = label
        self.address = address
        self.controls = controls
        self.params = params

    def __eq__(self, other: object) -> bool:
        return (self.label, self.address, self.controls, self.params) == (
        other.label, other.address, other.controls, self.params)

    def __str__(self) -> str:
        d = dict()
        d['gate'] = self.label
        d['address'] = self.address
        d['controls'] = self.controls
        d['params'] = self.params
        return d.__str__()

    def __repr__(self) -> str:
        return self.__str__()


class QuantumChannel:
    name: str
    gates: List[List[GateData]]
    errors: List[List[GateData]]
    probabilities: List[List[float]]

    def __init__(self, name=None, gates: List[List[GateData]] = None, errors: Dict = None, instructions=None,
                 probabilities: Dict = None, from_serializable=None) -> None:
        if from_serializable is None:
            self.name = name
            self.gates = gates
            self.errors = errors
            self.instructions = instructions
            self.probabilities = probabilities
        else:
            self.name = from_serializable['name']
            self.gates = []
            for g_seq in from_serializable['gates']:
                seq_obj = []
                for g in g_seq:
                    seq_obj.append(GateData(g['gate'], g['address'], g['controls'], g['params']))
                self.gates.append(seq_obj)

            self.errors = dict()
            for (error_id, error_seqs) in from_serializable['errors'].items():
                self.errors[error_id] = []

                for seq in error_seqs:
                    seq_obj = []
                    for g in seq:
                        seq_obj.append(GateData(g['gate'], g['address'], g['controls'], g['params']))
                    self.errors[error_id].append(seq_obj)
            self.instructions = from_serializable['instructions']
            self.probabilities = from_serializable['probabilities']
        assert len(self.gates) == len(self.probabilities.keys())
        # for (key, probs) in self.probabilities.items():
        #     assert len(probs) == len(self.errors[key]) + 1
        #     rel_tol = 1 / (10 ** (Precision.PRECISION + 1))
        #     assert isclose(sum(probs), 1.0, rel_tol=Precision.rel_tol)

    def __eq__(self, other: object) -> bool:
        if self.name != other.name:
            return False

        if self.gates != other.gates:
            return False

        if self.errors != other.errors:
            return False

        if self.instructions != other.instructions:
            return False

        if self.probabilities != other.probabilities:
            return False

        return True

    def dump_channel(self):
        d = dict()
        d['name'] = self.name
        d['gates'] = self.gates
        d['errors'] = self.errors
        d['instructions'] = self.instructions
        d['probabilities'] = self.probabilities
        return d

    def print_channel(self):
        print(f"***** {self.name} *****")
        for i in range(len(self.gates)):
            print("------")
            print(f"{self.probabilities[i][0]} ~~ {self.gates[i]}")
            for (j, err) in enumerate(self.errors[i]):
                print(f"{self.probabilities[i][j + 1]} ~~ {err}")

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.__str__()

    def get_uniform_prob_over_errors(self, error):
        if len(self.errors) == 0:
            result = []
        else:
            err_prob = error / len(self.errors)
            result = [err_prob for _ in self.errors]
        return result

    def get_addresses(self) -> List[int]:
        addresses = []
        for gate in self.gates[0]:
            assert gate.address is not None
            if not (gate.address in addresses):
                addresses.append(gate.address)
        return addresses
    
    def get_target(self) -> int:
        addresses =self.get_addresses()
        assert len(addresses) == 1
        return addresses[0] 

    def get_controls(self) -> List[int]:
        controls = []
        for op in self.gates[0]:
            if op.controls is None:
                if None not in controls:
                    controls.append(None)
            else:
                assert len(op.controls) == 1
                if op.controls[0] not in controls:
                    controls.append(op.controls[0])
        return controls
    
    def get_control(self) -> int:
        controls = self.get_controls()
        assert len(controls) == 1 or len(controls) == 0
        if len(controls) == 0:
            return None
        return controls[0]

    def get_params(self):
        return self.gates[0][0].params
    
    def get_instruction(self):
        assert len(self.instructions) == 1
        return self.instructions[0]


def is_there_none(c):
    if c is None:
        return True
    for i in c:
        if i is None:
            return True
    return False


def generate_channels_latex_spec(channels, name, inv_m, output_path=""):
    lines = []
    lines.append("\\begin{table}[h!]")
    lines.append("\t\\centering")
    lines.append("\t\\begin{tabular}{ |c|c|c| }")
    lines.append("\t\t\\hline")
    lines.append(f"\t\t instruction & error & probability \\\\")
    lines.append("\t\t\\hline")
    for channel in channels:
        assert isinstance(channel, QuantumChannel)
        channel_data = channel.dump_channel()
        channel_name = channel.name
        if Instruction.MEAS in channel.instructions:
            success_on0 = float(channel.probabilities[0][0])
            lines.append(f"\t\t{channel_name} & success on 0 & {success_on0}\\\\")
            success_on1 = float(channel.probabilities[1][0])
            lines.append(f"\t\t{channel_name} & success on 1 & {success_on1}\\\\")
        else:
            assert len(channel.errors.keys())  == 1
            assert 0 in channel.errors.keys()
            errors = channel.errors[0]
            probabilities = channel.probabilities[0]
            assert len(probabilities) -1 == len(errors)
            lines.append(f"\t\t{channel_name} & \identityMatrix & {float(probabilities[0])}\\\\")
            probabilities = probabilities[1:]
            for (error, prob) in zip(errors, probabilities):
                err_str = ""
                for err in error[1:]:
                    assert isinstance(err, GateData)
                    assert err.controls is None
                    if len(err_str) > 0 :
                        err_str += ", "
                    err_str += f"{err.label.name}{err.address}"
                lines.append(f"\t\t{channel_name} & {err_str} & {float(prob)}\\\\")
    lines.append("\t\t \\hline")
    lines.append("\t\\end{tabular}")
    channel_name = channel.name.replace("_", "-")
    lines.append("\t\\caption{Hardware specification for " + name + "}")
    lines.append("\t\\label{tab:config-" + name + "}")
    lines.append("\\end{table}\n")

    file = open(output_path + f"{name}.tex", "w")
    file.write("\n".join(lines))
    file.close()



class Vertex:
    counter = 0

    def __init__(self, quantum_state, classical_state, from_serializable=None, depth=0) -> None:
        if from_serializable is None:
            assert quantum_state is not None
            assert classical_state is not None
            self.id = Vertex.counter
            Vertex.counter += 1
            self.quantum_state = quantum_state
            self.classical_state = classical_state
            self.in_edges = dict()  # Channel -> Set[int]
            self.out_edges = dict()  # List of channels
            self.depth = depth
        else:
            self.id = from_serializable['id']
            self.quantum_state = quantum_state
            self.classical_state = classical_state
            self.in_edges = from_serializable['in_edges']
            self.out_edges = from_serializable['out_edges']
            self.depth = from_serializable['depth']

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, Vertex)
        return other.id == self.id

    def get_serializable(self):
        return {
            'id': self.id,
            'qs': self.quantum_state.sparse_vector,
            'cs': self.classical_state.sparse_vector,
            'in_edges': self.in_edges,
            'out_edges': self.out_edges,
            'depth': self.depth
        }


class Queue:
    queue: deque

    def __init__(self) -> None:
        self.queue = deque()

    def pop(self) -> Vertex:
        if self.is_empty():
            raise Exception("trying to pop empty queue")
        return self.queue.popleft()

    def push(self, v):
        self.queue.append(v)

    def is_empty(self) -> bool:
        return len(self.queue) == 0

    def len(self) -> int:
        return len(self.queue)


def get_random_number():
    precision = 10000000000000000
    a = random.randint(1, precision)
    return float(a) / float(precision)


def choose_from_prob_distribution(elements_: List[int], distribution_: List[float]):
    # random_num = get_random_number()
    # assert random_num >=0
    # assert random_num <= 1
    # # print(random_num, distribution, sum(distribution))
    # curr_top = 0

    elements = []
    distribution = []
    for (element, p_) in zip(elements_, distribution_):
        p = round(p_, Precision.PRECISION)
        if p > 0:
            elements.append(element)
            distribution.append(p)
        # if random_num <= curr_top + p or isclose(random_num, curr_top+p):
        #     return element
        # curr_top += p
    assert len(elements) > 0
    assert len(elements) == len(distribution)
    return random.choices(elements, weights=distribution)[0]


def choose_random(choices, weights_= (lambda x : 1)):
    weights = []
    for c in choices:
        weights.append(weights_(c))
    return random.choices(choices, weights=weights)[0]


def int_to_bin(n: int) -> str:
    assert n >= 0
    if n == 0:
        return "0"
    result = ""
    while n > 0:
        if (n % 2) == 1:
            result += "1"
        else:
            result += "0"
        n = floor(n / 2)
    return result

def get_mapped_basis(n, address_space):
    n_bin = int_to_bin(n)

    result = 0

    for (index, b) in enumerate(n_bin):
        if b == "1":
            result += 1 << address_space[index]

    return result



def int_to_bin2(n: int) -> str:
    bin_num = int_to_bin(n)
    return f"0b{bin_num}"


def get_vertices_observable(vertices, graph):
    o = None
    for v in vertices:
        if o is None:
            o = graph.v_to_observable[v]
        else:
            assert o == graph.v_to_observable[v]
    return o


def get_one(s: set):
    for e in s:
        return e


class Belief:
    d: Dict

    def __init__(self, d=None) -> None:
        if d is None:
            self.d = dict()
        else:
            self.d = d

    def get(self, v: int):
        if v in self.d.keys():
            return self.d[v]
        return 0.0

    def set_val(self, key, value, do_floor=False):
        assert key not in self.d.keys()
        if do_floor:
            if Precision.is_lowerbound:
                self.d[key] = myfloor(value, Precision.PRECISION)
            else:
                self.d[key] = myceil(value, Precision.PRECISION)
        else:
            self.d[key] = round(value, Precision.PRECISION)

    def add_val(self, key, value, do_floor=False):
        if do_floor:
            if Precision.is_lowerbound:
                value = myfloor(value, Precision.PRECISION)
            else:
                value = myceil(value, Precision.PRECISION)
        else:
            value = round(value, Precision.PRECISION)

        if key not in self.d.keys():
            self.set_val(key, value)
        else:
            self.d[key] += value
            if do_floor:
                if Precision.is_lowerbound:
                    self.d[key] = myfloor(self.d[key], Precision.PRECISION)
                else:
                    self.d[key] = myceil(self.d[key], Precision.PRECISION)
            else:
                self.d[key] = round(self.d[key], Precision.PRECISION)
    def get_sum(self):
        return sum(self.d.values())

    def check(self):
        rel_tol = 1 / (10 ** (Precision.PRECISION + 1))
        assert isclose(sum(self.d.values()), 1.0, rel_tol=Precision.rel_tol)

    def normalize(self,do_floor=False):
        sum_all = sum(self.d.values())
        for key in self.d.keys():
            if do_floor:
                if Precision.is_lowerbound:
                    self.d[key] = myfloor(self.d[key] / sum_all, Precision.PRECISION)
                else:
                    self.d[key] = myceil(self.d[key] / sum_all, Precision.PRECISION)
            else:
                self.d[key] = round(self.d[key] / sum_all, Precision.PRECISION)

    def __str__(self) -> str:
        return self.d.__str__()

    def __repr__(self):
        return str(self)

    def __eq__(self, other) -> bool:
        assert isinstance(other, Belief)
        rel_tol = 1 / (10 ** (Precision.PRECISION + 1))
        for (k, val) in self.d.items():
            other_val = other.get(k)
            if not isclose(val, other_val, rel_tol=Precision.rel_tol):
                return False
        for (k, val_) in other.d.items():
            self_val = round(self.get(k), Precision.PRECISION)
            val = round(val_, Precision.PRECISION)
            if not isclose(val, self_val, rel_tol=Precision.rel_tol):
                return False
        return True


def get_most_likely_vertices(b: Belief, heuristic=0):
    # b.check() # fix this overflows
    b.normalize()
    answer = set()
    if heuristic == 0:
        max_value = max(b.d.values())
        for (key, value) in b.d.items():
            if value == max_value:
                answer.add(key)
    else:
        assert heuristic == 1
        pop = []
        w = []
        for (key, val) in b.d.items():
            pop.append(key)
            w.append(val)
        res = random.choices(pop, w, k=1)[0]
        answer.add(res)
    return answer


def get_best_moves_on_belief(possible_deltas, adj_list, rankings, most_likely_vertices, current_obs):
    # to_higher = dict()
    # to_equal = dict()
    to_less = dict()

    for v in most_likely_vertices:
        # to_higher[v] = set()
        # to_equal[v] = set()
        to_less[v] = set()

    all_to_less = set()
    for delta in possible_deltas:
        for likely_vertex in most_likely_vertices:
            if likely_vertex in adj_list.keys():
                if current_obs in adj_list[likely_vertex].keys():
                    post_vertices = adj_list[likely_vertex][current_obs][delta]
                    post_v_rankings = [rankings[v][obs] for (obs, v) in post_vertices]
                    max_post_rank = max(post_v_rankings)
                    if max_post_rank < rankings[likely_vertex][current_obs]:
                        to_less[likely_vertex].add(delta)
                        all_to_less.add(delta)
            # elif max_post_rank == rankings[likely_vertex][current_obs]:
            #     to_equal[likely_vertex].add(delta)
            # else:
            #     to_higher[likely_vertex].add(delta)

    for (v, to_less_deltas) in to_less.items():
        all_to_less = all_to_less.intersection(to_less_deltas)
    if len(all_to_less) != 0:
        return list(all_to_less)
    return possible_deltas


def update_belief(current_obs, instruction, current_belief, obs_to_v, adj_probs):
    vertices = obs_to_v[current_obs]
    new_belief = Belief()
    assert isinstance(current_belief, Belief)
    for v in vertices:
        v_prob = current_belief.get(v)
        print()
        post_vertices = adj_probs[current_obs][v][instruction]
        for (pv, pv_prob) in post_vertices.items():
            new_belief.add_val(pv, v_prob * pv_prob)
    return new_belief

def normalize_belief_after_meas(obs, obs_to_v, current_belief):
    sum_ = 0.0
    new_belief = Belief()
    vertices_in_obs = obs_to_v[obs]

    for (v, prob) in current_belief.d.items():
        if v in vertices_in_obs:
            new_belief.set_val(v, prob)
            sum_ += prob
    if sum_ > 0:
        for v in new_belief.d.keys():
            new_belief.d[v] /= sum_
            new_belief.d[v] = round(new_belief.d[v], Precision.PRECISION)
    else:
        # TODO: Investigate this case
        for v in new_belief.d.keys():
            new_belief.d[v] = 1 / len(new_belief.d.keys())
            new_belief.d[v] = round(new_belief.d[v], Precision.PRECISION)
    return new_belief


class Instruction(enum.Enum):
    # PAULI GATES
    X = "X"
    Y = "Y"
    Z = "Z"
    I = "I"

    # PHASE GATES
    T = "T"
    TD = "TD"
    S = "S"
    SD = "SD"
    H = 'H'
    SX = "SX"
    SXD = "SXD"

    U1 = "U1"
    U2 = "U2"
    U3 = "U3"

    # MULTIQUBIT GATES
    CNOT = "CNOT"
    CU3 = "CU3"
    CZ = "CZ"

    # MEASUREMENT (in Z axis)
    MEAS = "MEAS"
    MEASX = "MEASX"

    # NON-UNITARY
    RESET = "RESET"

    def __repr__(self) -> str:
        return self.__str__()
    def __eq__(self, other):
        return self.name == other.name
    
def instruction_to_ibm(qc, basis_gates, instruction, target, control=None):
    assert isinstance(instruction, Instruction)
    assert isinstance(qc, QuantumCircuit)
    if instruction == Instruction.X:
        # if 'x' in basis_gates:
        qc.x(target)
        # else:
        #     assert 'u3' in basis_gates
        #     qc.u(np.pi, 0, np.pi, target)
    elif instruction == Instruction.Z:
        if 'z' in basis_gates:
            qc.z(target)
        else:
            assert 'u1' in basis_gates
            qc.u(0,0, np.pi)
    elif instruction == Instruction.H:
        # if 'h' in basis_gates:
        qc.h(target)
        # else:
        #     qc.u(np.pi/2, 0, np.pi, target)
    elif instruction == Instruction.MEAS:
        qc.measure(target, target)
    elif instruction == Instruction.CNOT:
        assert 'cx' in basis_gates
        assert control is not None
        qc.cx(control, target)
    elif instruction == Instruction.I:
        pass
    else:
        raise Exception(f"Instruction {instruction.name} could not be translated to IBM instruction. Missing implementation.")



def instruction_to_gate(instruction: Instruction) -> Gate:
    if instruction == Instruction.X:
        return Gate.X
    if instruction == Instruction.Z:
        return Gate.Z
    if instruction == Instruction.Y:
        return Gate.Y
    if instruction == Instruction.I:
        return Gate.I
    if instruction == Instruction.T:
        return Gate.T
    if instruction == Instruction.TD:
        return Gate.TD
    if instruction == Instruction.S:
        return Gate.S
    if instruction == Instruction.SD:
        return Gate.SD
    if instruction == Instruction.SX:
        return Gate.SX
    if instruction == Instruction.SXD:
        return Gate.SXD
    if instruction == Instruction.H:
        return Gate.H
    if instruction == Instruction.CNOT:
        return Gate.CNOT
    if instruction == Instruction.CU3:
        return Gate.CU3
    if instruction == Instruction.CZ:
        return Gate.CZ
    if instruction == Instruction.U3:
        return Gate.U3
    if instruction == Instruction.U2:
        return Gate.U2
    if instruction == Instruction.U1:
        return Gate.U1
    if instruction == Instruction.RESET:
        return Gate.RESET
    if (instruction == Instruction.MEAS) or (instruction == Instruction.MEASX):
        return None

    raise Exception(f"Cannot translate instruction ({instruction}) into gate ")


class Result:
    """Synthesised algorithm return this result.
    """
    n_instructions: int  # number of instructions that the simulator executed
    qmemory: Any  # final quantum state of the qpu
    meas_cache: Any  # final classical state of qpu
    log: List[str]  # log of the qpu, for debugging purposes
    belief: Belief  # last belief before termination of the algorithm
    observable: int  # last observable before termiantion of the algorithm

    def __init__(self, n_instructions, qmemory, meas_cache, log, belief, observable, applied_ins, done=False,
                 error=False) -> None:
        self.n_instructions = n_instructions
        self.qmemory = qmemory
        self.meas_cache = meas_cache
        self.log = log
        self.belief = belief
        self.observable = observable
        self.done = done
        self.instructions_applied = applied_ins
        self.error = error


def write_csv_results(df, path):
    df.pivot_table(values='accuracy',
                   index=df.error_prob,
                   columns='algorithm',
                   aggfunc='first').to_csv(path)


def default_mapping(num_qubits=10000) -> Dict[int, int]:
    result = {}
    for i in range(num_qubits):
        result[i] = i
    return result

def are_all_instructions_id(ops: List[Instruction]) -> bool:
    for op in ops:
        assert isinstance(op, Instruction)
        if op != Instruction.I:
            return False
    return True


def does_ops_contains_err(ops: List[Instruction], err: Instruction) -> bool:
    for op in ops:
        assert isinstance(op, Instruction)
        if op == err:
            return True
    return False


def algorithm_exists(all_algos, new_algo):
    for algo in all_algos:
        if algo == new_algo:
            return True
    return False


def create_channel(instruction: Instruction, prob_success: float, target: int, control: int, errors,
                   prob_errors: List[float], params=None) -> QuantumChannel:
    assert len(prob_errors) == len(errors)
    probabilities = dict()
    probabilities[0] = [prob_success] + prob_errors
    error_dict = dict()

    gate = instruction_to_gate(instruction)

    error_gates = []
    for err in errors:
        ins_err = [GateData(gate, target, controls=[control], params=None)]
        for (e, t) in err:
            assert isinstance(e, Instruction)
            ins_err.append(GateData(instruction_to_gate(e), t, None, None))
        error_gates.append(ins_err)
    error_dict[0] = error_gates

    if control is not None:
        name = f"{instruction.name}{target}_{control}"
    else:
        name = f"{instruction.name}{target}"

    return QuantumChannel(name, [[GateData(gate, target, [control])]], error_dict, [instruction],
                          probabilities=probabilities)


channelMEAS = QuantumChannel("MEAS", [[Gate.P0], [Gate.P1]], {0: [], 1: []}, [Instruction.MEAS], {0: [1.0], 1: [1.0]})

# ----------- MERGE CHANNELS FUNCTIONS ------------

def clean_previous_instructions(target, current_sequence):
    # we remove all current_instruction that involve target because there is a reset after them
    new_sequence = []

    for err in current_sequence:
        assert isinstance(err, GateData)
        assert err.controls is None
        assert err.label in [Gate.RESET, Gate.X, Gate.Z, Gate.Y]
        if err.address != target:
            new_sequence.append(err)
    return new_sequence

def insert_x_gate(err_sequence, gate, target):
    assert gate.label == gate.X
    result = []
    foundX = False
    for err in err_sequence:
        assert isinstance(err, GateData)
        assert err.controls is None
        assert err.label in [Gate.RESET, Gate.X, Gate.Z, Gate.Y]
        if foundX:
            result.append(err)
        else:
            if err.address == target:
                if err.label == Gate.X:
                    foundX = True 
                elif err.label == Gate.Y:
                    foundX = True
                    result.append(GateData(Gate.Z, err.address, err.controls, err.params))
                else:
                    result.append(err)
            else:
                result.append(err)

    if not foundX:
        result.append(gate)
    return result

def insert_z_gate(err_sequence, gate, target):
    assert gate.label == Gate.Z
    result = []
    foundZ = False
    for err in err_sequence:
        assert isinstance(err, GateData)
        assert err.controls is None
        assert err.label in [Gate.RESET, Gate.X, Gate.Z, Gate.Y]
        if foundZ:
            result.append(err)
        else:
            if err.address == target:
                if err.label == Gate.Z:
                    foundZ = True 
                elif err.label == Gate.Y:
                    foundZ = True
                    result.append(GateData(Gate.X, err.address, err.controls, err.params))
                else:
                    if err.label == Gate.RESET:
                        # Z gate does not have effect when a qubit is the computational basis
                        foundZ = True
                    result.append(err)
            else:
                result.append(err)
    if not foundZ:
        result.append(gate)
    return result

def insert_y_gate(err_sequence, gate, target):
    result = insert_x_gate(err_sequence, GateData(Gate.X, target, gate.controls, gate.params), target)
    result = insert_z_gate(result, GateData(Gate.Z, target, gate.controls, gate.params), target)
    return result

def clean_instruction_err_seq(err_sequence):
    qubits_on_reset = set()
    new_err_sequence = []
    for err in err_sequence:
        assert isinstance(err, GateData)
        gate = err.label
        target = err.address
        assert err.controls is None
        assert err.params is None
        assert gate in [Gate.RESET, Gate.X, Gate.Z, Gate.Y]
        if gate == Gate.RESET:
            qubits_on_reset.add(target)
            new_err_sequence = clean_previous_instructions(target, new_err_sequence)
            new_err_sequence.append(err)
        elif gate == Gate.X:
            new_err_sequence = insert_x_gate(new_err_sequence, err, target)
        elif gate == Gate.Z:
            new_err_sequence = insert_z_gate(new_err_sequence, err, target)
        elif gate == Gate.Y:
            new_err_sequence = insert_y_gate(new_err_sequence, err, target)
        else:
            raise Exception("This should not be reachable")
        new_err_sequence.append(err)
    return new_err_sequence

def remove_low_probs(l, probs):
    assert len(l) + 1 == len(probs)
    result = []
    result_probs = [probs[0]]
    for (index, p_) in enumerate(probs[1:]):
        if Precision.is_lowerbound:
            p = myfloor(p_, Precision.PRECISION)
        else:
            p = myceil(p_, Precision.PRECISION)
        if not isclose(p, 0, abs_tol=Precision.isclose_abstol):
            result.append(l[index])
            result_probs.append(p)
    return result, result_probs

def merge_channels(channel1: QuantumChannel, channel2: QuantumChannel):
    assert isinstance(channel1, QuantumChannel)
    assert isinstance(channel2, QuantumChannel)

    gates1 = channel1.gates
    gates2 = channel2.gates

    if len(gates1) != 1 or len(gates2) != 1:
        raise Exception("Missing implementation.")

    good_seq = [gates1[0] + gates2[0]]
    instructions = channel1.instructions + channel2.instructions

    probabilities = [channel1.probabilities[0][0] * channel2.probabilities[0][
        0]]  # this is the probability that the right case happens

    errors1 = channel1.errors[0]

    errors2 = channel2.errors[0]

    errors = []

    for (i2, err2) in enumerate(errors2):
        errors.append(gates1[0] + err2)
        probabilities.append(channel1.probabilities[0][0] * channel2.probabilities[0][i2 + 1])

    for (i1, err1) in enumerate(errors1):
        errors.append(err1 + gates2[0])
        probabilities.append(channel2.probabilities[0][0] * channel1.probabilities[0][i1 + 1])

    for (i1, err1) in enumerate(errors1):
        prob1 = channel1.probabilities[0][i1 + 1]
        for (i2, err2) in enumerate(errors2):
            prob2 = channel2.probabilities[0][i2 + 1]
            err_sequence = err1[1:] + err2[1:]
            final_error_sequence  = [err1[0], err2[0]] + clean_instruction_err_seq(err_sequence)
            if final_error_sequence not in errors:
                errors.append(final_error_sequence)
                probabilities.append(prob1 * prob2)
            else:
                for (index, err) in enumerate(errors):
                    if err == final_error_sequence:
                        probabilities[index] += (prob1 * prob2)
                        break

    errors, probabilities = remove_low_probs(errors, probabilities)
    if not isclose(sum(probabilities), 1.0, rel_tol=Precision.rel_tol):
        print(f"WARNING: channel probabilities sum to {float(sum(probabilities))}")
    return QuantumChannel("MC", gates=good_seq, errors={0: errors}, instructions=instructions,
                          probabilities={0: probabilities})

def simplify_channel(qc: QuantumChannel):
    assert len(qc.errors) == 1


class AlgorithmNode:
    instructions: Instruction

    def __init__(self, instruction=None, target=None, control=None, params=None, next_ins=None, case0=None, case1=None,
                 count_meas=0, serialized=None) -> None:
        if serialized is None:
            assert isinstance(target, int)
            assert isinstance(control, int) or (control is None)
            assert instruction is not None
            assert isinstance(instruction, Instruction)
            self.instruction = instruction
            self.target = target
            self.control = control
            self.params = params
            self.next_ins = next_ins
            self.case0 = case0
            self.case1 = case1
            self.count_meas = count_meas
            self.depth = 0
            if self.next_ins is not None:
                self.depth = self.next_ins
            if case1 is not None:
                assert case0 is not None
                self.depth = max(case1.depth, case0.depth)
            if instruction != Instruction.I:
                self.depth += 1
        else:
            assert instruction is None
            assert target is None
            assert control is None
            assert params is None
            assert next_ins is None
            assert case0 is None
            assert case1 is None
            self.instruction = serialized['instruction']
            self.target = serialized['target']
            self.control = serialized['control']
            self.params = serialized['params']
            # self.count_meas = serialized['count_meas']
            self.count_meas = 0
            # self.depth = serialized['depth']
            self.depth = 0

            if serialized['next'] is None:
                self.next_ins = None
                if serialized['case0'] is not None:
                    self.case0 = AlgorithmNode(serialized=serialized['case0'])
                else:
                    self.case0 = None

                if serialized['case1'] is not None:
                    self.case1 = AlgorithmNode(serialized=serialized['case1'])
                else:
                    self.case1 = None
            else:
                assert serialized['case0'] is None
                assert serialized['case1'] is None
                self.next_ins = AlgorithmNode(serialized=serialized['next'])
                self.case0 = None
                self.case1 = None

    def __eq__(self, other: object) -> bool:
        if other is None:
            return False
        return (self.instruction == other.instruction) and (self.target == self.target) and (
                    self.control == self.control) and (self.params == self.params) and (self.next_ins == other.next_ins) and (self.case0 == other.case0) and (self.case1 == other.case1)
    
    def get_instructions_used(self, current_set) -> List[Instruction]:
        current = []
        if self.instruction != Instruction.I:
            current = (self.instruction, self.target, self.control)
            if current not in current_set:
                current_set.append(current)
        
        if self.next_ins is not None:
            assert self.case0 is None
            assert self.case1 is None
            return self.next_ins.get_instructions_used(current_set)
        else:
            if self.case0 is not None:
                self.case0.get_instructions_used(current_set)
            if self.case1 is not None:
                self.case1.get_instructions_used(current_set)
    
    def serialize(self, for_json=False):
        if for_json:
            next_ins = -1
        else:
            next_ins = None
        if self.next_ins is not None:
            next_ins = self.next_ins.serialize(for_json)
        
        if for_json:
            case0 = -1
            case1 = -1
        else:
            case0 = None
            case1 = None
        if self.case0 is not None:
            case0 = self.case0.serialize(for_json)
        if self.case1 is not None:
            case1 = self.case1.serialize(for_json)

        params = self.params
        if (params is None) and for_json:
            params = -1
        
        control = self.control
        if (control is None) and for_json:
            control = -1

        instruction = self.instruction
        if for_json:
            instruction = instruction.name
        return {
            "instruction": instruction,
            "target": self.target,
            "control": control,
            "params": params,
            "next": next_ins,
            "case0": case0,
            "case1": case1,
            "count_meas": self.count_meas,
            "depth": self.depth
        }
    
# Parsing Kraus operators
def decompose_matrix(m) -> Dict[str, float]:
    assert m.shape[0] == m.shape[1]
    assert m.shape[0] == 2
    m = np.matrix(m)
    a = m[0,0]
    b = m[0,1]
    c = m[1,0]
    d = m[1,1]

    alpha0 = (a+d)/2
    alpha1 = (b+c)/2
    alpha2 = a - alpha0
    alpha3 = (alpha1-b) * (-I)
    if Precision.is_lowerbound:
        result = {
            'ID' : myfloor(alpha0*conjugate(alpha0), Precision.PRECISION),
            'X' : myfloor(alpha1*conjugate(alpha1), Precision.PRECISION),
            'Z' : myfloor(alpha2*conjugate(alpha2), Precision.PRECISION),
            'Y' : myfloor(alpha3*conjugate(alpha3), Precision.PRECISION)
        }
    else:
        result = {
            'ID' : myceil(alpha0*conjugate(alpha0), Precision.PRECISION),
            'X' : myceil(alpha1*conjugate(alpha1), Precision.PRECISION),
            'Z' : myceil(alpha2*conjugate(alpha2), Precision.PRECISION),
            'Y' : myceil(alpha3*conjugate(alpha3), Precision.PRECISION)
        }
    sum_ = sum(result.values())
    # if not isclose(sum_, 1.0, rel_tol=Precision.rel_tol):
    #     print(f"kraus operators probabilities do not sum 1: {float(sum_)}")
    for (key, val) in result.items():
        if Precision.is_lowerbound:
            result[key] = myfloor(val/sum_, Precision.PRECISION)
        else:
            result[key] = myceil(val/sum_, Precision.PRECISION)
    assert isclose(sum(result.values()), 1.0, rel_tol=Precision.rel_tol)
    return result

def get_matrix_from_coeffs(coeffs):
    I = np.array([[1, 0], [0, 1]])
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    Y = np.array([[0 ,-I], [I, 0]])

    new_matrix = coeffs["I"]*I + coeffs["X"]*X + coeffs["Z"]*Z + coeffs["Y"]*Y
    return new_matrix

def get_kraus_matrix(instruction):
    matrix = None
    for m in instruction["params"]:
        if matrix is None:
            matrix = m
        else:
            matrix += m
    return matrix

def get_instruction(key, instruction):
    assert isinstance(key, str)
    assert key == "X" or key == "Z" or key == "ID" or key == "Y"
    op = key.lower()
    return {"name": op, "qubits": instruction["qubits"]}

meas_instructions = [Instruction.MEAS, Instruction.MEASX]

def ibm_traditional1(qc: QuantumCircuit,  basis_gates, cr: ClassicalRegister):
    instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 2, 1)
    instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 2, 0)
    instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
    with qc.if_test((cr[2], 1)):
        instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
    

def ibm_traditional2(qc: QuantumCircuit,  basis_gates, cr: ClassicalRegister):
    instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 2, 1)
    instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 2, 0)
    instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
    with qc.if_test((cr[2], 1)) as else0_:
        instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
    with else0_:
        instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
        with qc.if_test((cr[2], 1)):
            instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
    

def ibm_traditional3(qc: QuantumCircuit,  basis_gates, cbits: ClassicalRegister):
    instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 2, 1)
    instruction_to_ibm(qc, basis_gates, Instruction.CNOT, 2, 0)
    instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
    with qc.if_test((cbits[2], 0)) as else0_:
        instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
        with qc.if_test((cbits[2], 0)) as else1_:
            pass
        with else1_:
            instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
            with qc.if_test((cbits[2], 0)) as else2_:
                pass
            with else2_:
                instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
                pass
    with else0_:
        instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
        with qc.if_test((cbits[2], 0)) as else1_:
            instruction_to_ibm(qc, basis_gates, Instruction.MEAS, 2, None)
            with qc.if_test((cbits[2], 0)) as else2_:
                pass
            with else2_:
                instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)
                pass
        with else1_:
            instruction_to_ibm(qc, basis_gates, Instruction.X, 0, None)

def load_algorithms_file(path) -> List[AlgorithmNode]:
    mod = SourceFileLoader("m", path).load_module()
    result = []
    for algorithm in mod.algorithms:
        if algorithm is not None:
            result.append(AlgorithmNode(serialized=algorithm))
    return result

def execute_algorithm(node: AlgorithmNode, qpu, for_ibm, count_ins=0, basis_gates=None, cbits=None, address_space=None):
    if node is None:
        return count_ins
    if for_ibm:
        assert basis_gates is not None
        assert cbits is not None
        instruction_to_ibm(qpu, basis_gates, node.instruction, node.target, node.control)
        assert address_space is None
        if node.next_ins is not None:
            assert node.case0 is None
            assert node.case1 is None
            return execute_algorithm(node.next_ins, qpu, for_ibm, count_ins+1, basis_gates=basis_gates, cbits=cbits)
        elif node.instruction in meas_instructions:
            assert node.instruction in meas_instructions
            with qpu.if_test((cbits[node.target], 0)) as else0_:
                execute_algorithm(node.case0, qpu, for_ibm, count_ins+1, basis_gates=basis_gates, cbits=cbits)
            with else0_:
                execute_algorithm(node.case1, qpu, for_ibm, count_ins+1, basis_gates=basis_gates,cbits=cbits)
            return 1
    else:
        assert address_space is not None
        if node.control is None:
            control_ = None
        else:
            control_ = address_space[node.control]
        outcome = qpu.apply_instructions([node.instruction], [address_space[node.target]] , [control_])
        if node.next_ins is not None:
            assert node.case0 is None
            assert node.case1 is None
            return execute_algorithm(node.next_ins, qpu, for_ibm, count_ins+1, basis_gates=basis_gates, address_space=address_space)
        else:
            # assert node.instruction in meas_instructions
            if outcome == 0:
                return execute_algorithm(node.case0, qpu, for_ibm, count_ins+1, basis_gates=basis_gates, address_space=address_space)
            else:
                return execute_algorithm(node.case1, qpu, for_ibm, count_ins+1, basis_gates=basis_gates, address_space=address_space)
            
def is_unitary(U):
    dagger = U.getH()
    res = U * dagger
    rows = res.shape[0]
    cols = res.shape[1]
    assert rows == cols
    for i in range(rows):
        for j in range(cols):
            if i == j:
                if not isclose(res[i,j], 1, rel_tol=Precision.rel_tol):
                    return False
            else:
                if not isclose(res[i,j], 0, abs_tol=Precision.isclose_abstol):
                    return False
                
    return True

def get_channel_algorithm_node(channel: QuantumChannel, inverse_mapping):
    assert len(channel.instructions) > 0
    if channel.get_control() is not None:
        c = inverse_mapping[channel.get_control()]
    else:
        c = None

    head = AlgorithmNode(channel.get_instruction(), inverse_mapping[channel.get_target()], c, channel.get_params())
    current = head
    gate_sequence = channel.gates[0]
    assert isinstance(gate_sequence, list)
    for (index, instruction) in channel.instructions[1:]:
        assert instruction not in meas_instructions
        gate_data = gate_sequence[index]
        assert isinstance(gate_data, GateData)
        target = gate_data.address
        assert len(gate_data.controls) == 1
        control = gate_data.controls[0]
        params = gate_data.params
        next_node = AlgorithmNode(instruction, target, control, params=params)
        current.next_ins = next_node
        current = next_node

    return head, current

class Guard:
    def __init__(self, guard, target_channels: List[QuantumChannel]) -> None:
        self.guard = guard
        self.target_channels = target_channels

def partial_trace(rho, dims, keep):
    """
    Compute the partial trace of a density matrix over specified subsystems.
    
    Args:
    - rho (np.ndarray): The density matrix of the composite system.
    - dims (list): Dimensions of each subsystem.
    - keep (list): List of indices of subsystems to keep.
    
    Returns:
    - np.ndarray: The reduced density matrix after tracing out the specified subsystems.
    """
    # Total number of subsystems
    num_subsystems = len(dims)
    
    # Permute the dimensions so that the subsystems to keep come first
    perm = keep + [i for i in range(num_subsystems) if i not in keep]
    perm_dims = [dims[i] for i in perm]
    perm_rho = rho.reshape(perm_dims + perm_dims).transpose(perm + [num_subsystems + i for i in perm])
    
    # Trace out the subsystems to be traced out
    for i in range(num_subsystems - len(keep)):
        perm_rho = np.trace(perm_rho, axis1=-1, axis2=num_subsystems - len(keep))
    
    return perm_rho




        

