from copy import deepcopy

from utils import *
from typing import Set, List
from qiskit_aer.noise import NoiseModel as IBMNoiseModel
from qiskit.extensions import HGate, XGate, ZGate, CXGate
from itertools import permutations 
from qiskit.visualization import plot_gate_map

# used to prepare initial states
noisefree_h = HGate(label="H1")
noisefree_x = XGate(label="X1")
noisefree_z = ZGate(label="Z1")
noisefree_cx = CXGate(label="CX1")


class NoiseData:
    target_instructions: Set[Instruction]
    apply_instructions: List[List[Instruction]]
    probabilities: List[float]
    qubits: Set[int]
    controls: Set[int]

    def __init__(self, target_instructions, apply_instructions, probabilities, qubits, params, controls=None) -> None:
        assert len(apply_instructions) == len(probabilities)
        self.target_instructions = target_instructions # noise only affects these instructions
        self.apply_instructions = apply_instructions # instructions to apply with probability i
        self.probabilities = probabilities # apply with probability i apply_instructions[i]
        self.qubits = qubits # only apply this noise to the qubits in this list
        self.controls = controls
        self.params = params
        # rel_tol = 1/(10**(Precision.PRECISION-1))
        # if isinstance(self.probabilities[0], list):
        #     for prob in self.probabilities:
        #         if not isclose(sum(prob), 1, rel_tol=Precision.rel_tol):
        #             raise Exception("Probabilities of noise data should equal 1")
        # else:
        #     print(sum(self.probabilities))
        #     if not isclose(sum(self.probabilities), 1, rel_tol=Precision.rel_tol):
        #         raise Exception("Probabilities of noise data should equal 1")

    def get_success_probability(self):
        temp = []
        temp_ins = []
        for (index, instruction) in enumerate(self.apply_instructions):
            if is_identity(instruction) or self.probabilities[index] > 0.5:
                temp_ins.append(instruction)
            # if are_all_instructions_id(instruction):
                if Precision.is_lowerbound:
                    temp.append(float(myfloor(self.probabilities[index], Precision.PRECISION)))
                else:
                    temp.append(float(myceil(self.probabilities[index], Precision.PRECISION)))
        if len(temp) == 0:
            temp.append(0.0)
        assert len(temp) == 1
        return temp[0]
        # raise Exception("No success probability found")
    
    def get_err_probability(self, err: Instruction):
        probs = 0.0
        for (index, instruction) in enumerate(self.apply_instructions):
            if does_ops_contains_err(instruction, err) or does_ops_contains_err(instruction, Instruction.Y):
                probs += self.probabilities[index]
        if Precision.is_lowerbound:
            return myfloor(probs, Precision.PRECISION)
        else:
            return myceil(probs, Precision.PRECISION)

    def is_single_qubit_gate_noise(self):
        if Instruction.CNOT in self.target_instructions:
            return False
        return True
                
def get_noisyCNOT_model(prob_err):
    
    readout_errors = []
    unitary_errors = []
    readouterr = NoiseData([Instruction.MEAS, Instruction.MEASX], apply_instructions=[Instruction.X, Instruction.I], probabilities=[[0, 1], [0, 1]], qubits=[2], params=None)
    readout_errors.append(readouterr)

    noise_datacx = NoiseData([Instruction.CNOT],
                                 [[Instruction.I], [Instruction.X]],[1-prob_err, prob_err], qubits=[2],params=[[2, 2], [2,2]], controls=[0])
    unitary_errors.append(noise_datacx)

    noise_datacx = NoiseData([Instruction.CNOT],
                                 [[Instruction.I]],[1], qubits=[2],params=[[1, 1]], controls=[1])
    unitary_errors.append(noise_datacx)

    noise_datax = NoiseData([Instruction.X, Instruction.Z, Instruction.H], [
            [Instruction.I]], probabilities=[1],qubits=[0], params=[[0,0]])
    unitary_errors.append(noise_datax)
    noise_datax = NoiseData([Instruction.X, Instruction.Z, Instruction.H], [
            [Instruction.I]], probabilities=[1],qubits=[2], params=[[2,2]])
    unitary_errors.append(noise_datax)
    
    
    noise_model = NoiseModel(None, hardware=None, readout_error=readout_errors, name="faulty_cnot", basis_gates=['x', 'cx', 'measure'])
    noise_model.noise = unitary_errors
    return noise_model

class NoiseModel:
    noise: List[NoiseData]
    readout_error: List[NoiseData]
    name: str
    def __init__(self, ibm_noise_model, hardware, readout_error=None, name="", basis_gates=None) -> None:
        self.ibm_noise_model = ibm_noise_model
        self.hardware = hardware
        self.noise = []
        self.readout_error = readout_error
        assert isinstance(name, str)
        self.name = name
        if self.ibm_noise_model is not None:
            self.basis_gates = self.ibm_noise_model.basis_gates
        else:
            self.basis_gates = basis_gates

    def plot_topology(self):
        return plot_gate_map(self.hardware)

    def get_basis_gates(self):
        return self.basis_gates
    
    def get_num_qubits(self):
        try:
            return self.hardware.configuration().num_qubits
        except:
            return self.hardware.num_qubits
        
    def get_qubit_indegree(self, target):
        result = 0
        for noise in self.noise:
            if Instruction.CNOT in noise.target_instructions:
                if target in noise.qubits:
                    result += 1
        return result
    
    def dump_model(self, path, filename, target_qubits, target_ins_, f=-1):
        def contains_qubits(l, qubits):
            if l is None:
                return False
            for q in qubits:
                if q in l:
                    return True
            return False
        
        file = open(path+filename, "w")
        file.write("controls,target,error_seq,target_ins,probability\n")
        for n in self.noise:
            if (contains_qubits(n.controls, target_qubits) or contains_qubits(n.qubits, target_qubits)) and contains_qubits(n.target_instructions, target_ins_):
                target_ins = []
                for ti in n.target_instructions:
                    assert isinstance(ti, Instruction)
                    target_ins.append(ti.name)
                target_ins = "|".join(target_ins)

                assert isinstance(n, NoiseData)
                for (index, l_ins) in enumerate(n.apply_instructions):
                    parsed_ins = []
                    for i in l_ins:
                        parsed_ins.append(i.name)
                    if f == -1:
                        p = n.probabilities[index]
                    else:
                        assert f > 0
                        p = round(n.probabilities[index], f)
                    if p > 0:
                        if n.controls is None:
                            ctls = ""
                        else:
                            ctls = "|".join([str(c) for c in n.controls])
                        parsed_ins = "|".join(i + " " + str(param) for (i, param) in  zip(parsed_ins, n.params[index]))
                        assert len(n.qubits) == 1
                        file.write(f"{ctls},{n.qubits[0]},{parsed_ins},{target_ins},{p}\n")
        file.close()

        file = open(path+"re_"+filename, "w")
        file.write("target,read0,read0_err,read1,read1_err\n")
        for n in self.readout_error:
            assert len(n.probabilities) == 2
            assert len(n.probabilities[0]) == len(n.probabilities[1])
            assert len(n.probabilities[0]) == 2
            if contains_qubits(n.qubits, target_qubits):
                probs_str = ""
                for i in range(2):
                    for j in range(2):
                        if f == -1:
                            p = n.probabilities[i][(j+1)%2]
                        else:
                            p = round(n.probabilities[i][(j+1)%2], f)
                        if len(probs_str) > 0:
                            probs_str += ","
                        probs_str += str(p)
                assert len(n.qubits) == 1
                
                file.write(f"{n.qubits[0]},{probs_str}\n")  
        file.close()

    def add_noise(self, noise: NoiseData):
        self.noise.append(noise)

    def get_most_noisy_qubits(self, top=-1) -> List[int]:
        assert top != 0
        result = []
        for noise in self.noise:
            assert isinstance(noise, NoiseData)
            if noise.is_single_qubit_gate_noise():
                assert len(noise.qubits) == 1
                qubit = noise.qubits[0]
                result.append((noise.get_success_probability(),[i.name for i in noise.target_instructions],qubit))

        result = sorted(result, key=lambda x : x[0], reverse=False)
        if top == -1:
            return result
        return result[:top]
    
    def get_most_noisy_couplers(self, top=-1) -> List[int]:
        assert top != 0
        result = []
        for noise in self.noise:
            assert isinstance(noise, NoiseData)
            if not noise.is_single_qubit_gate_noise():
                assert len(noise.qubits) == 1
                assert len(noise.controls) == 1
                result.append((noise.get_success_probability(), [noise.qubits[0], noise.controls[0]]))
        result = sorted(result, key=lambda x : x[0], reverse=False)
        if top == -1:
            return result
        return result[:top]

    def get_qubit_couplers(self, target) -> List[int]:
        assert (target >= 0)
        result = []

        for noise in self.noise:
            if not noise.is_single_qubit_gate_noise():
                assert len(noise.qubits) == 1
                assert len(noise.controls) == 1
                if target == noise.qubits[0]:
                    result.append([noise.controls[0], float(noise.get_success_probability())])

        result = sorted(result, key=lambda x : x[1], reverse=False)
        return result
    
    def get_available_couplers(self) -> List[int]:
        result = []
        for noise in self.noise:
            assert isinstance(noise, NoiseData)
            if not noise.is_single_qubit_gate_noise():
                assert len(noise.qubits) == 1
                assert len(noise.controls) == 1
                result.append([noise.qubits[0], noise.controls[0]])
        return result
    
    def is_valid_embedding(self, embedding, needed_couplers) -> bool:
        mapping = {}
        for (hn1, hn2) in embedding:
            for (n1, n2) in needed_couplers:
                if n1 not in mapping.keys():
                    mapping[n1] = hn1
                else:
                    if mapping[n1] != hn1:
                        return None
                if n2 not in mapping.keys():
                    mapping[n2] = hn2
                else:
                    if mapping[n2] != hn2:
                        return None
        return mapping
        
    def get_all_embeddings(self, num_qubits, needed_couplers, max_embeddings=10) -> List[Dict[int, int]]:
        couplers = self.get_available_couplers()
        
        valid_embeddings = []
        for permutation in permutations(couplers, num_qubits):
            
            res = self.is_valid_embedding(permutation, needed_couplers)
            if res is not None:
                valid_embeddings.append(res)

            if len(valid_embeddings) == max_embeddings:
                return valid_embeddings
            
        assert valid_embeddings <= max_embeddings
        for (index, e) in enumerate(valid_embeddings):
            for e2 in valid_embeddings[index+1:]:
                assert e != e2
        return valid_embeddings
            


    
    def get_most_noisy_readout(self, top=-1) -> List[int]:
        if self.readout_error is None:
            raise Exception("No readout noise model")
        assert top != 0
        result = []
        for noise in self.readout_error:
            assert isinstance(noise, NoiseData)
            assert len(noise.qubits) == 1
            assert len(noise.probabilities) == 2
            success_probability = (noise.probabilities[0][1] + noise.probabilities[1][1])/2
            if Precision.is_lowerbound:
                success_probability = myfloor(success_probability, Precision.PRECISION)
            else:
                success_probability = myceil(success_probability, Precision.PRECISION)
            result.append((success_probability, noise.qubits[0]))
        result = sorted(result, key=lambda x : x[0], reverse=False)
        if top == -1:
            return result
        return result[:top]
    
    def get_qubit_error(self, target):
        assert isinstance(target, int)
        result = []
        for noise in self.noise:
            assert isinstance(noise, NoiseData)
            if noise.is_single_qubit_gate_noise():
                assert len(noise.qubits) == 1
                qubit = noise.qubits[0]
                if qubit == target:
                    result.append(noise.get_success_probability())

        if len(result) == 0:
            print("WARNING: no single qubit error model found")
        return result
    
    def get_qubit_noise_obj(self, target):
        assert isinstance(target, int)
        result = []
        for noise in self.noise:
            assert isinstance(noise, NoiseData)
            if noise.is_single_qubit_gate_noise():
                assert len(noise.qubits) == 1
                qubit = noise.qubits[0]
                if qubit == target:
                    result.append(noise)

        if len(result) == 0:
            print("WARNING: no single qubit error model found")
        return result
    
    
    
    def insert_new_err(self, seq, new_ins):
        if new_ins[0] == Instruction.Y:
            answer = self.insert_new_err(seq, (Instruction.X, new_ins[1]))
            answer = self.insert_new_err(answer, (Instruction.Z, new_ins[1]))
            return answer
        answer = []
        # print("WARNING: SHOULD FIX INSERT NEW ERR")
        # raise Exception("Fix me before running more experiments")
        # if new_ins[0] in [s[0] for s in seq]:
        #     for s in seq:
        #         answer.append(s)
        #     return answer
        seq.append(new_ins)
        return seq

    def simplify_err_seq(self, apply_ins_seq, params):
        error_seq = []
        for (index, ins) in enumerate(apply_ins_seq):
            if ins != Instruction.I:
                error_seq = self.insert_new_err(error_seq, (ins, params[index]))
        if len(error_seq) > 0:
            return error_seq
        return None

    def get_meas_channel(self, instruction, target):
        noise_obj =  self.get_qubit_readout_error(target)
        if len(noise_obj) != 1:
            raise Exception("The number of readout error models are more than one.")
        noise_obj = noise_obj[0]

        probabilities = dict()
        if Precision.is_lowerbound:
            probabilities[0] = [myfloor(p, Precision.PRECISION) for p in noise_obj.probabilities[0][::-1]]
            probabilities[1] = [myfloor(p, Precision.PRECISION) for p in noise_obj.probabilities[1][::-1]]
        else:
            probabilities[0] = [myceil(p, Precision.PRECISION) for p in noise_obj.probabilities[0][::-1]]
            probabilities[1] = [myceil(p, Precision.PRECISION) for p in noise_obj.probabilities[1][::-1]]

        error_dict = dict()
        if instruction == Instruction.MEAS:
            gates = [
                [GateData(Gate.P0, target, None, None), GateData(Gate.WRITE0, target, None, None)], [GateData(Gate.P1, target, None, None), GateData(Gate.WRITE1, target, None, None)]
                ]
            error_dict[0] = [[GateData(Gate.P0, target, None, None), GateData(Gate.WRITE1, target, None, None)]]
            error_dict[1] = [[GateData(Gate.P1, target, None, None), GateData(Gate.WRITE0, target,      None, None)]]
        else:
            raise Exception("FIX probabilities")
            assert instruction == Instruction.MEASX
            gates = [
                [GateData(Gate.H, target, None, None), GateData(Gate.P0, target, None, None), GateData(Gate.H, target, None, None), GateData(Gate.WRITE0, target, None, None)], [GateData(Gate.H, target, None, None), GateData(Gate.P1, target, None, None), GateData(Gate.H, target, None, None), GateData(Gate.WRITE1, target, None, None)]
                ]
            
            error_dict[0] = [
                [GateData(Gate.H, target, None, None), GateData(Gate.P1, target, None, None), GateData(Gate.H, target, None, None), GateData(Gate.WRITE0, target, None, None)]
                ]
            error_dict[1] = [
                [GateData(Gate.H, target, None, None), GateData(Gate.P0, target, None, None), GateData(Gate.H, target, None, None), GateData(Gate.WRITE1, target, None, None)]
                ]

        return QuantumChannel(f"{instruction.name}{target}", 
                              gates=gates,
                              errors=error_dict,
                              instructions=[instruction],
                              probabilities=probabilities)
    def get_noise_object(self, control, target, instruction):
        if control is None:
            noise_objects = self.get_qubit_noise_obj(target)
        else:
            noise_objects = self.get_coupler_noise_obj(control, target)

        noise_object = None

        for noise_obj in noise_objects:
            assert isinstance(noise_obj, NoiseData)
            if instruction in noise_obj.target_instructions:
                noise_object = noise_obj
                break


        if noise_object is None:
            raise Exception("There are no noise objects for this physical qubit")
        return noise_object

    def get_instruction_channel(self, target: int, instruction: Instruction, control=None, params=None):
        """_summary_

        Args:
            target (int): Qubit in noise model
            instruction (Instruction): _description_
        """
        if control is None:
            if instruction in [Instruction.MEAS, Instruction.MEASX]:
                return self.get_meas_channel(instruction, target)
            noise_objects = self.get_qubit_noise_obj(target)
        else:
            assert instruction == Instruction.CNOT
            noise_objects = self.get_coupler_noise_obj(control, target)
        noise_object = None
        # First, we find the specific noise object we want to work with
        for noise_obj in noise_objects:
            assert isinstance(noise_obj, NoiseData)
            if instruction in noise_obj.target_instructions:
                noise_object = noise_obj
                break

        if noise_object is None:
            raise Exception("Cannot build instruction channel if there are no noise objects for this physical qubit")
        
        errors = []
        prob_errors = []
        success_prob = None
        found = False
        for (index, seq) in enumerate(noise_object.apply_instructions):
            if Precision.is_lowerbound:
                prob = myfloor(noise_object.probabilities[index], Precision.PRECISION)
            else:
                prob = myceil(noise_object.probabilities[index], Precision.PRECISION)
        
            if is_identity(seq):
                assert found == False
                assert success_prob is None
                success_prob = prob
                found = True
            else:
                err_seq = self.simplify_err_seq(seq, noise_object.params[index])
                if err_seq is not None:
                    errors.append(err_seq)
                    prob_errors.append(prob)
        # assert success_prob is not None
        if success_prob is None:
            success_prob = 0
        return create_channel(instruction, success_prob, target, control, errors, prob_errors, params=params)
    
    def get_coupler_noise_obj(self, control, target):
        assert isinstance(control, int)
        assert isinstance(target, int)

        result = []
        for noise in self.noise:
            assert isinstance(noise, NoiseData)
            if not noise.is_single_qubit_gate_noise():
                if (control in noise.controls) and (target in noise.qubits):
                    result.append(noise)
        if len(result) == 0:
            print(f"WARNING: no noise model found for couplers {control} {target}")
        return result

    
    def get_avg_qubit_error(self):
        result = []
        for noise in self.noise:
            assert isinstance(noise, NoiseData)
            if noise.is_single_qubit_gate_noise():
                result.append(noise.get_success_probability())

        if len(result) == 0:
            print("WARNING: no single qubit error model found")
        return sum(result)/len(result)

    def get_coupler_error(self, target, control):
        assert isinstance(target, int)
        assert isinstance(control, int)
        result = []
        for noise in self.noise:
            assert isinstance(noise, NoiseData)
            if not noise.is_single_qubit_gate_noise():
                assert len(noise.qubits) == 1
                assert len(noise.controls) == 1
                if noise.qubits[0] == target and noise.controls[0] == control:
                    result.append((noise.target_instructions, noise.get_success_probability()))
        if len(result) == 0:
            print(f"WARNING: no noise model found for coupler {target} {control}")
        return result
    
    def get_avg_coupler_error(self):
        result = []
        for noise in self.noise:
            assert isinstance(noise, NoiseData)
            if not noise.is_single_qubit_gate_noise():
                result.append(noise.get_success_probability())
        if len(result) == 0:
            print(f"WARNING: no noise model found for couplers")
        return sum(result)/len(result)
    
    def get_avg_coupler_zerror(self):
        result = []
        for noise in self.noise:
            assert isinstance(noise, NoiseData)
            if not noise.is_single_qubit_gate_noise():
                result.append(noise.get_err_probability(Instruction.Z))
        if len(result) == 0:
            print(f"WARNING: no noise model found for couplers")
        return sum(result)/len(result)
    
    def get_avg_coupler_xerror(self):
        result = []
        for noise in self.noise:
            assert isinstance(noise, NoiseData)
            if not noise.is_single_qubit_gate_noise():
                result.append(noise.get_err_probability(Instruction.X))
        if len(result) == 0:
            print(f"WARNING: no noise model found for couplers")
        return sum(result)/len(result)
    
    def get_qubit_readout_error(self, target):
        if self.readout_error is None:
            raise Exception("No readout noise model")
        assert isinstance(target, int)
        result = []
        for noise in self.readout_error:
            assert isinstance(noise, NoiseData)
            assert len(noise.qubits) == 1
            assert len(noise.probabilities) == 2
            if noise.qubits[0] == target:
                result.append(noise)

        if len(result) == 0:
            print(f"No readout error found for qubit {target}")

        if len(result) > 1:
            print(f"WARNING: more than one model for readout error found for {target}")
        return result
    
    def get_avg_readout_error(self):
        if self.readout_error is None:
            raise Exception("No readout noise model")
        result = []
        for noise in self.readout_error:
            assert isinstance(noise, NoiseData)
            assert len(noise.qubits) == 1
            assert len(noise.probabilities) == 2
            success_probability = (noise.probabilities[0][1] + noise.probabilities[1][1])/2
            if Precision.is_lowerbound:
                success_probability = myfloor(success_probability, Precision.PRECISION)
            else:
                success_probability = myceil(success_probability, Precision.PRECISION)
            result.append(success_probability)

        if len(result) == 0:
            print(f"No readout error found")

        return sum(result)/len(result)
    
    def get_z_error_prob(self, target, target_instruction: Instruction, control=None) -> float:
        probs = []
        for noise in self.noise:
            if target_instruction in noise.target_instructions:
                assert noise.qubits == -1 or len(noise.qubits) == 1
                if noise.qubits == -1 or noise.qubits[0] == target:
                    if (noise.controls == -1) or (control is None) or (control in noise.controls):
                        probs.append(noise.get_err_probability(Instruction.Z))
        assert len(probs) == 1
        return probs[0]
    
    def get_x_error_prob(self, target, target_instruction: Instruction, control=None) -> float:
        probs = []
        for noise in self.noise:
            if target_instruction in noise.target_instructions:
                assert noise.qubits == -1 or len(noise.qubits) == 1
                if noise.qubits == -1 or noise.qubits[0] == target:
                    if (noise.controls == -1) or (control is None) or (control in noise.controls):
                        probs.append(noise.get_err_probability(Instruction.X))
        assert len(probs) == 1
        return probs[0]
    
    def add_qubit_error(self, target, instructions, errorx, errorz, errory, reseterr, measerr0, measerr1):
        assert errorx + errorz + errory + reseterr < 1.0
        assert measerr0 < 1.0
        assert measerr1 < 1.0

        success_probability = 1 - (errorx + errorz + errory + reseterr)

        # create readout_error data
        read_noise_data = NoiseData(
            [Instruction.MEAS, Instruction.MEASX], 
            [Instruction.X, Instruction.I], 
            [[measerr0, 1-measerr0],[measerr1, 1-measerr1]], 
            [target],
            [target, target])

        if self.readout_error is None:
            self.readout_error = []
        self.readout_error.append(read_noise_data)

        # create instructions noise model
        apply_instructions = [
            [Instruction.I],
            [Instruction.X],
            [Instruction.Z],
            [Instruction.Y],
            [Instruction.RESET]
        ]
        
        ins_noise_data = NoiseData(
            instructions,
            apply_instructions,
            [success_probability, errorx, errorz, errory, reseterr],
            [target],
            params=[[target], [target], [target], [target], [target]]
        )
        self.noise.append(ins_noise_data)

    def add_coupler_error(self, target, cls, instructions, errorx, errorz, errory, reseterr):
        assert isinstance(cls, int)
        params = []
        apply_instructions = []
        probabilities = []
        ins = [Instruction.X, Instruction.Z, Instruction.Y, Instruction.RESET, Instruction.I]
        p_ins = [errorx, errorz, errory, reseterr, 1 - errorx - errorz - errory - reseterr]
        for (index1, i1) in enumerate(ins):
            for (index2, i2) in enumerate(ins):
                p = p_ins[index1] * p_ins[index2]
                if p > 0:
                    params.append([target, cls])
                    apply_instructions.append([i1, i2])
                    probabilities.append(p)

        ins_noise_data = NoiseData(
            instructions,
            apply_instructions,
            probabilities,
            [target],
            params=params,
            controls=[cls]
        )
        self.noise.append(ins_noise_data)
 

def get_read_out_error(prob_error, qubits=-1):
    noise_data = NoiseData([Instruction.MEAS, Instruction.MEASX], 
                     [Instruction.X, Instruction.I], 
                     [[prob_error, 1-prob_error], [prob_error, 1-prob_error]],
                     qubits=qubits, params=[qubits, qubits])
    return noise_data


def translate_op(op):
    if op == 'id':
        return [Instruction.I]
    if op == 'cx' or op =='ecr':
        return [Instruction.CNOT]
    if op == 'x':
        return [Instruction.X, Instruction.Z]
    if op == 'sx':
        return [Instruction.SX, Instruction.U1, Instruction.H]
    if op == 'reset':
        return [Instruction.RESET]
    if op == 'u2':
        return [Instruction.U2, 
                Instruction.H, # U2(0, pi)
                ]
    if op == 'u1':
        return [Instruction.U1]
    if op == 'u3':
        return [Instruction.U3, Instruction.X,
                Instruction.Z]
    if op == "cz":
        return [Instruction.CZ]
    raise Exception(f"Not possible to translate operation {op}")

def translate_ibm_instructions(error_instructions, qubits=-1):
    apply_instructions = []
    params = []
    for instructions in error_instructions:
        current_list = []
        current_params = []
        for instruction in instructions:
            if instruction['name'] == 'id':
                current_list.append(Instruction.I)
                if qubits == -1:
                    current_params.append(-1)
                else:
                    current_params.append(qubits[instruction['qubits'][0]])
                assert len(instruction['qubits']) == 1
            elif instruction['name'] == 'z':
                current_list.append(Instruction.Z)
                if qubits == -1:
                    current_params.append(-1)
                else:
                    current_params.append(qubits[instruction['qubits'][0]])
                assert len(instruction['qubits']) == 1
            elif instruction['name'] == 'reset':
                current_list.append(Instruction.RESET)
                if qubits == -1:
                    current_params.append(-1)
                else:
                    current_params.append(qubits[instruction['qubits'][0]])
                assert len(instruction['qubits']) == 1
            elif instruction['name'] == 'x':
                current_list.append(Instruction.X)
                if qubits == -1:
                    current_params.append(-1)
                else:
                    current_params.append(qubits[instruction['qubits'][0]])
                assert len(instruction['qubits']) == 1
            elif instruction['name'] == 'y':
                current_list.append(Instruction.Y)
                if qubits == -1:
                    current_params.append(-1)
                else:
                    current_params.append(qubits[instruction['qubits'][0]])
                assert len(instruction['qubits']) == 1
            elif instruction['name'] == 'pauli':
                assert len(instruction['params']) == 1
                if instruction['params'][0] == "II":
                    current_list.append(Instruction.I)
                    current_list.append(Instruction.I)
                elif instruction['params'][0] == "IX":
                    current_list.append(Instruction.I)
                    current_list.append(Instruction.X)
                elif instruction['params'][0] == "XI":
                    current_list.append(Instruction.X)
                    current_list.append(Instruction.I)
                elif instruction['params'][0] == "XX":
                    current_list.append(Instruction.X)
                    current_list.append(Instruction.X)
                elif instruction['params'][0] == "XZ":
                    current_list.append(Instruction.X)
                    current_list.append(Instruction.Z)
                elif instruction['params'][0] == "XY":
                    current_list.append(Instruction.X)
                    current_list.append(Instruction.Y)
                elif instruction['params'][0] == "IY":
                    current_list.append(Instruction.I)
                    current_list.append(Instruction.Y)
                elif instruction['params'][0] == "YI":
                    current_list.append(Instruction.Y)
                    current_list.append(Instruction.I)
                elif instruction['params'][0] == "YY":
                    current_list.append(Instruction.Y)
                    current_list.append(Instruction.Y)
                elif instruction['params'][0] == "YX":
                    current_list.append(Instruction.Y)
                    current_list.append(Instruction.X)
                elif instruction['params'][0] == "YZ":
                    current_list.append(Instruction.Y)
                    current_list.append(Instruction.Z)
                elif instruction['params'][0] == "IZ":
                    current_list.append(Instruction.I)
                    current_list.append(Instruction.Z)
                elif instruction['params'][0] == "ZI":
                    current_list.append(Instruction.Z)
                    current_list.append(Instruction.I)
                elif instruction['params'][0] == "ZZ":
                    current_list.append(Instruction.Z)
                    current_list.append(Instruction.Z)
                elif instruction['params'][0] == "ZX":
                    current_list.append(Instruction.Z)
                    current_list.append(Instruction.X)
                elif instruction['params'][0] == "ZY":
                    current_list.append(Instruction.Z)
                    current_list.append(Instruction.Y)
                else:
                    raise Exception(f"unimplemented translation of Pauli {instruction['params']}")
                assert len(instruction['qubits']) == 2
                if qubits == -1:
                    current_params.append(-1)
                    current_params.append(-1)
                else:
                    current_params.append(qubits[instruction['qubits'][0]])
                    current_params.append(qubits[instruction['qubits'][1]])
            else:
                raise Exception(f"Instruction not valid {instruction['name']}")
                return None, None
        apply_instructions.append(current_list)
        params.append(current_params)
    assert len(params) == len(apply_instructions)
    return apply_instructions, params

def simplify_qubit_instructions(instructions):
    ''' Remove extra identity gates, and redudant gates such as applying a unitary U followed by its inverse
    '''
    assert len(instructions[0].keys()) == 2
    assert 'qubits' in instructions[0].keys()
    assert 'name' in instructions[0].keys()
    answer = []
    target = instructions[0]['qubits'][0]
    answer.append({'name': 'id', 'qubits': [target]})
    contains_reset = False
    for instruction in instructions:
        name = instruction['name']
        if name == 'reset':
            contains_reset = True
            answer = []
        assert name in ['reset', 'id', 'z', 'x', 'y']
        # we assume that the sequence of instructions apply to the same qubit always
        assert len(instruction['qubits']) == 1
        assert instruction['qubits'][0] == target
        if name != 'id':

            if name !='reset' and name == answer[-1]['name']:
                answer = answer[:-1]
            elif name == 'y':
                answer.append({'name': 'x', 'qubits': [target]})
                answer.append({'name': 'z', 'qubits': [target]})
            else:
                answer.append(instruction)
            assert name != 'pauli'
    return answer, contains_reset

def get_instruction_unitary(instruction):
    assert isinstance(instruction, str)
    if instruction == 'x':
        u = [[0,1], [1,0]]
    elif instruction == 'z':
        u = [[1, 0], [0, -1]]
    elif instruction == 'id':
        u = [[1,0], [0,1]]
    else:
        raise Exception(f"cannot convert {instruction} to unitary.")
    return np.matrix(u)

def get_instructions_unitary(instructions):
    assert len(instructions[0].keys()) == 2
    assert 'qubits' in instructions[0].keys()
    assert 'name' in instructions[0].keys()
    result = get_instruction_unitary("id")
    for instruction in instructions:
        name = instruction['name']
        assert name in ['x', 'z', 'id', 'y']
        ins_u = get_instruction_unitary(name)
        result = result * ins_u
    assert is_unitary(result)
    return result

def reduce_qubit_noise_model(error_instructions, probabilities):
    new_error_instructions = []
    new_probabilities = []
    common_errors = dict()
    common_errors['id'] = 0.0
    common_errors['x'] = 0.0
    common_errors['y'] = 0.0
    common_errors['z'] = 0.0

    for (index, instructions) in enumerate(error_instructions):
        simplified_instructions, contains_reset = simplify_qubit_instructions(instructions)
        assert isinstance(simplified_instructions, list)
        if contains_reset:
            new_error_instructions.append(simplified_instructions)
            new_probabilities.append(probabilities[index])
        else:
            unitary = get_instructions_unitary(simplified_instructions)
            assert is_unitary(unitary)
            decomposed = decompose_matrix(unitary)
            assert len(decomposed.keys()) == 4
            for (key, val) in decomposed.items():
                assert isinstance(key, str)
                common_errors[key.lower()] += probabilities[index]*val
    target = error_instructions[0][0]['qubits'][0]
    for (key, val) in common_errors.items():
        new_error_instructions.append([{'name': key.lower(), 'qubits': [target]}])
        new_probabilities.append(val)
    assert len(new_error_instructions) == len(new_probabilities)
    return new_error_instructions, new_probabilities

def get_pauli_params_instruction(s, target):
    assert isinstance(s, str)
    assert len(s) == 1
    qubits = [target]
    if s == "I":
        name = "id"
    else:
        assert s in ["X", "Y", "Z"]
        name = s.lower()
    return {'name': name, 'qubits': qubits}


def simplify_cx_instructions(instructions):
    assert isinstance(instructions, list)
    ins0 = []
    ins1 = []
    contains_reset = False
    for instruction in instructions:
        name = instruction['name']
        if name == 'pauli':
            assert len(instruction['params']) == 1
            params = instruction['params'][0]
            assert len(params) == 2
            
            assert isinstance(params, str)
            assert instruction['qubits'][0] == 0 # if this fail change below
            assert instruction['qubits'][1] == 1
            ins0.append(get_pauli_params_instruction(params[0], 0))
            ins1.append(get_pauli_params_instruction(params[1], 1))
        else:
            assert name in ['id', 'x', 'y', 'z', "reset"]
            if name == "reset":
                contains_reset = True
            assert len(instruction['qubits']) == 1
            target = instruction['qubits'][0]
            if target == 0:
                ins0.append(instruction)
            else:
                assert target == 1
                ins1.append(instruction)
    r0, contains_reset0 = simplify_qubit_instructions(ins0)
    r1, contains_reset1 = simplify_qubit_instructions(ins1)
    assert contains_reset == contains_reset0 or contains_reset1
    return r0, r1, contains_reset


def reduce_cx_noise_model(error_instructions, probabilities):
    pauli_errs = ["X", "Y", "Z", "I"]

    common_errors = dict()
    for e0 in pauli_errs:
        for e1 in pauli_errs:
            err = f"{e0}{e1}"
            assert err not in common_errors.keys()
            common_errors[err] = 0.0

    new_error_instructions = []
    new_probabilities = []

    for (index, instructions) in enumerate(error_instructions):
        ins0, ins1, contains_reset = simplify_cx_instructions(instructions)
        if contains_reset:
            new_ins = ins0 + ins1
            f = False
            for (index2, ins)  in enumerate(new_error_instructions):
                if new_ins == ins:
                    f = True
                    new_probabilities[index2] += probabilities[index]

                    break
            if not f:
                new_error_instructions.append(new_ins)
                new_probabilities.append(probabilities[index])
        else:
            unitary0= get_instructions_unitary(ins0)
            unitary1 = get_instructions_unitary(ins1)
            assert is_unitary(unitary0)
            assert is_unitary(unitary1)
            decomposed0 = decompose_matrix(unitary0)
            decomposed0['I'] = decomposed0["ID"]
            decomposed1 = decompose_matrix(unitary1)
            decomposed1['I'] = decomposed1["ID"]
            p = probabilities[index]
            for err0 in pauli_errs:
                val0 = decomposed0[err0]
                for err1 in pauli_errs:
                    val1 = decomposed1[err1]
                    common_errors[f"{err0}{err1}"] += p*val0*val1
                    assert (common_errors[f"{err0}{err1}"] < 1.0 or isclose())

    
    for (err, val) in common_errors.items():
        assert isinstance(err, str)
        assert len(err) == 2
        new_instructions = [get_pauli_params_instruction(err[0], 0), get_pauli_params_instruction(err[1], 1)]
        new_error_instructions.append(deepcopy(new_instructions))
        new_probabilities.append(val)
    return new_error_instructions, new_probabilities




def parse_kraus_error(error_seqs, probabilities):
    assert len(error_seqs) == len(probabilities)
    seqs_to_probs = []
    for (index, seq) in enumerate(error_seqs):
        current_seqs = [([], probabilities[index])]
        c = 0
        for instruction in seq:
            if instruction['name'] != "kraus":
                for (s, _) in current_seqs:
                    s.append(instruction)
            else:
                c+=1
                # assert c == 1
                matrix = get_kraus_matrix(instruction)
                current_seqs_ = []
                errors = decompose_matrix(matrix)
                for (s, p) in current_seqs:
                    cp_s = deepcopy(s)
                    for (key, val) in errors.items():
                        cp_s.append(get_instruction(key, instruction))
                        if val > 0:
                            current_seqs_.append((cp_s, p*val))
                current_seqs = current_seqs_
        seqs_to_probs.extend(current_seqs)

    result_errors = []
    result_probabilities = []
    for (s, p) in seqs_to_probs:
        result_errors.append(s)
        result_probabilities.append(p)
    return result_errors, result_probabilities


def translate_qerror(error):
    assert len(error['operations']) == 1
    target_ops = translate_op(error['operations'][0])
    qubits = error['gate_qubits'][0]
    if  (Instruction.CNOT in target_ops) or (Instruction.CZ in target_ops):
        # this is a 2-qubit gate
        assert len(qubits) == 2
        controls = [qubits[0]]
        target = [qubits[1]]
    else:
        assert len(qubits) == 1
        controls = None
        target = [qubits[0]]
    assert len(error['instructions']) == len(error['probabilities'])
    error_instructions, probabilities = parse_kraus_error(error['instructions'], error['probabilities'])
    assert len(error_instructions) == len(probabilities)
    
    if not ((Instruction.CNOT in target_ops) or (Instruction.CZ in target_ops)):
        error_instructions, probabilities = reduce_qubit_noise_model(error_instructions, probabilities)
    else:
        error_instructions, probabilities = reduce_cx_noise_model(error_instructions, probabilities)
    apply_instructions, params = translate_ibm_instructions(error_instructions, qubits)
    if apply_instructions is None:
        return None
    assert params is not None
    assert len(params) == len(apply_instructions)
    return NoiseData(target_ops, apply_instructions, probabilities, target, controls=controls, params=params)

def translate_readout_error(error):
    assert error['type'] == 'roerror'
    assert len(error['operations']) == 1
    assert error['operations'][0] == 'measure'
    assert len(error['probabilities']) == 2
    assert len(error['probabilities'][0]) == len(error['probabilities'][1])
    assert len(error['probabilities'][0]) == 2
    probabilities0 = error['probabilities'][0]
    temp = probabilities0[0]
    probabilities0[0] = probabilities0[1]
    probabilities0[1] = temp
    assert sum(probabilities0) == 1
    probabilities1 = error['probabilities'][1]
    assert sum(probabilities1) == 1
    assert len(error['gate_qubits']) == 1
    qubit = error['gate_qubits'][0]
    assert len(qubit) == 1
    qubit = qubit[0]
    params = [qubit, qubit]
    assert isinstance(qubit, int)
    return NoiseData([Instruction.MEAS, Instruction.MEASX], 
                     [Instruction.X, Instruction.I], 
                     [probabilities0, probabilities1],
                     [qubit], params=params)

def translate_readout_errors(errors):
    result = []
    qubits = set()
    for error in errors:
        noise_data = translate_readout_error(error)
        qubit = noise_data.qubits[0]
        assert qubit not in qubits
        qubits.add(qubit)
        result.append(noise_data)
    return result

def get_ibm_noise_model(backend_, for_ibm=False, path_prefix="", with_updated_meas_noise=False, noise_csv=None, thermal_relaxation=False) -> NoiseModel:
    """ returns a noise model 

    Args:
        backend_: a string of the noise model
        for_ibm: if for ibm a noise for ibm's quantum simulator is returned
    Returns:
        NoiseModel: 
    """
    assert isinstance(for_ibm, bool)
    assert isinstance(path_prefix, str)
    from qiskit.providers.fake_provider import FakeTenerife, FakeJohannesburg, FakeJakarta, FakePerth, FakeLagos, FakeNairobi, FakeHanoi, FakeCairo, FakeMumbai, FakeKolkata, FakePrague, FakeSherbrooke, FakeAlmaden, FakeArmonk, FakeAthens, FakeAuckland, FakeBelem, FakeBoeblingen, FakeBogota, FakeBrooklyn, FakeBurlington, FakeCambridge, FakeCasablanca, FakeEssex, FakeGeneva, FakeGuadalupe, FakeLima, FakeLondon, FakeManhattan, FakeManila, FakeMelbourne, FakeMontreal, FakeOslo, FakeOurense, FakeParis, FakeQuito, FakePoughkeepsie, FakeRochester, FakeRome, FakeRueschlikon, FakeSantiago, FakeSingapore, FakeSydney, FakeTokyo, FakeToronto, FakeVigo, FakeWashington, FakeYorktown

    if backend_ == FAKE_TENERIFE:
        backend = FakeTenerife()
    elif backend_ == FAKE_JOHANNESBURG:
        backend = FakeJohannesburg()
    elif backend_ == FAKE_PERTH:
        backend = FakePerth()
    elif backend_ == FAKE_LAGOS:
        backend = FakeLagos()
    elif backend_ == FAKE_NAIROBI:
        backend = FakeNairobi()
    elif backend_ ==  FAKE_HANOI:
        backend = FakeHanoi()
    elif backend_ == FAKE_CAIRO:
        backend = FakeCairo()
    elif backend_ == FAKE_MUMBAI:
        backend = FakeMumbai()
    elif backend_ == FAKE_KOLKATA:
        backend = FakeKolkata()
    elif backend_ == FAKE_PRAGUE:
        backend = FakePrague()
    elif backend_ == FAKE_SHERBROOKE:
        backend = FakeSherbrooke()
    elif backend_ == FAKE_ALMADEN:
        backend = FakeAlmaden()
    elif backend_ == FAKE_ARMONK:
        backend = FakeArmonk()
    elif backend_ == FAKE_ATHENS:
        backend = FakeAthens()
    elif backend_ == FAKE_AUCKLAND:
        backend = FakeAuckland()
    elif backend_ == FAKE_BELEM:
        backend = FakeBelem()
    elif backend_ == FAKE_BOEBLINGEN:
        backend = FakeBoeblingen()
    elif backend_ == FAKE_BOGOTA:
        backend = FakeBogota()
    elif backend_ == FAKE_BROOKLYN:
        backend = FakeBrooklyn()
    elif backend_ == FAKE_BURLINGTON:
        backend = FakeBurlington()
    elif backend_ == FAKE_CAMBRIDGE:
        backend = FakeCambridge()
    elif backend_ == FAKE_CASABLANCA:
        backend = FakeCasablanca()
    elif backend_ == FAKE_ESSEX:
        backend = FakeEssex()
    elif backend_ == FAKE_GENEVA:
        backend = FakeGeneva()
    elif backend_ == FAKE_GUADALUPE:
        backend = FakeGuadalupe()
    elif backend_ == FAKE_LIMA:
        backend = FakeLima()
    elif backend_ == FAKE_LONDON:
        backend = FakeLondon()
    elif backend_ == FAKE_MANHATTAN:
        backend = FakeManhattan()
    elif backend_ == FAKE_MANILA:
        backend = FakeManila()
    elif backend_ == FAKE_MELBOURNE:
        backend = FakeMelbourne()
    elif backend_ == FAKE_MONTREAL:
        backend = FakeMontreal()
    elif backend_ == FAKE_OSLO:
        backend = FakeOslo()
    elif backend_ == FAKE_OURENSE:
        backend = FakeOurense()
    elif backend_ == FAKE_JAKARTA:
        backend = FakeJakarta()
    elif backend_ == FAKE_PARIS:
        backend = FakeParis()
    elif backend_ == FAKE_QUITO:
        backend = FakeQuito()
    elif backend_ == FAKE_POUGHKEEPSIE:
        backend = FakePoughkeepsie()
    elif backend_ == FAKE_ROCHESTER:
        backend = FakeRochester()
    elif backend_ == FAKE_ROME:
        backend = FakeRome()
    elif backend_ == FAKE_RUESCHLIKON:
        backend = FakeRueschlikon()
    elif backend_ == FAKE_SANTIAGO:
        backend = FakeSantiago()
    elif backend_ == FAKE_SINGAPORE:
        backend = FakeSingapore()
    elif backend_ == FAKE_SYDNEY:
        backend = FakeSydney()
    elif backend_ == FAKE_TOKYO:
        backend = FakeTokyo()
    elif backend_ == FAKE_TORONTO:
        backend = FakeToronto()
    elif backend_ == FAKE_VIGO:
        backend = FakeVigo()
    elif backend_ == FAKE_WASHINGTON:
        backend = FakeWashington()
    elif backend_ == FAKE_YORKTOWN:
        backend = FakeYorktown()
    elif backend_ == FAKE_JAKARTA:
        backend = FakeJakarta()
    elif backend_ == FAKE_CUSCO:
        return parse_platform_csv(backend_, path_prefix, None, noise_csv)
    else:
        raise Exception("Cannot handle backend")

    hardware = backend
    try:
        backend_name = backend.name()
    except:
        backend_name = backend.name

    ibm_noise_model = IBMNoiseModel.from_backend(backend, thermal_relaxation=thermal_relaxation)
    if for_ibm:
        return ibm_noise_model
    
    backend = IBMNoiseModel.from_backend(backend, thermal_relaxation=thermal_relaxation).to_dict()['errors']
    ibm_readout_errors = []
    ibm_qerrors = []
    for error in backend:
        if error['type'] == 'qerror':
            ibm_qerrors.append(error)
        elif error['type'] == 'roerror':
            ibm_readout_errors.append(error)
        else:
            raise Exception(f"Error type {error['type']} unknown")
    if not with_updated_meas_noise:
        noise_model = NoiseModel(ibm_noise_model, hardware, translate_readout_errors(ibm_readout_errors), backend_)
    else:
        noise_model = NoiseModel(ibm_noise_model, hardware, parse_platform_csv(backend_, path_prefix, 1, noise_csv), backend_) 

    for error in ibm_qerrors:
        noise_data = translate_qerror(error)
        if noise_data is not None:
            noise_model.add_noise(noise_data)

    return noise_model

def is_identity(seq):
    for s in seq:
        assert(isinstance(s, Instruction))
        if s != Instruction.I:
            return False
    return True

def parse_platform_csv(backend, path_prefix="", prev_noise_model=None, noise_model_name=None):
    if noise_model_name is None:
        file = open(f"{path_prefix}noise_models/ibm_{backend}.csv")
    else:
        file = open(f"{path_prefix}noise_models/{noise_model_name}.csv")

    lines = file.readlines()[1:]

    readout_errors = []
    unitary_errors = []
    for line in lines:
        elements = line.split(",")
        qubit = int(elements[0])
        prob0_error = float(elements[6])
        prob1_error = float(elements[7])
        readouterr = NoiseData([Instruction.MEAS, Instruction.MEASX], apply_instructions=[Instruction.X, Instruction.I], probabilities=[[prob0_error, 1-prob0_error], [prob1_error, 1-prob1_error]], qubits=[qubit], params=None)
        readout_errors.append(readouterr)

        if prev_noise_model is None:
            # unitary errors
            probx_error = float(elements[11])
            noise_datax = NoiseData([Instruction.X, Instruction.Z], [
                [Instruction.I], [Instruction.X], [Instruction.Z], [Instruction.X, Instruction.Z]], probabilities=[1-probx_error, probx_error/3, probx_error/3, probx_error/3],qubits=[qubit], params=[[qubit, qubit], [qubit, qubit], [qubit, qubit], [qubit, qubit]])
            
            unitary_errors.append(noise_datax)
            noise_datacx = NoiseData([Instruction.CNOT],
                                    [[Instruction.I]],[1], qubits=[qubit],params=[[qubit, 900]], controls=[900])
            
            noise_datacx2 = NoiseData([Instruction.CNOT],
                                    [[Instruction.I]],[1], qubits=[qubit], params=[[qubit, 901]], controls=[901])
            
            unitary_errors.append(noise_datacx)
            unitary_errors.append(noise_datacx2)

        # missing create cx channel
    if prev_noise_model is None:
        noise_data900 = NoiseData([Instruction.X, Instruction.Z, Instruction.H], [
                [Instruction.I]], probabilities=[1],qubits=[900], params=[[900, 900]])
        noise_data901 = NoiseData([Instruction.X, Instruction.Z, Instruction.H], [
                [Instruction.I]], probabilities=[1],qubits=[901], params=[[901, 901]])
        unitary_errors.append(noise_data900)
        unitary_errors.append(noise_data901)
        noise_model = NoiseModel(None, hardware=None, readout_error=readout_errors, name=backend, basis_gates=['x', 'cx', 'measure'])
        noise_model.noise = unitary_errors
        return noise_model
    else:
        return readout_errors
    

