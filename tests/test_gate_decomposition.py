from qiskit import transpile, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RYGate, CPhaseGate, CSXGate
from qiskit_aer import AerSimulator
from numpy import pi
import os, sys


sys.path.append(os.getcwd()+"/..")
from qpu_utils import BasisGates
from ibm_noise_models import get_ibm_noise_model, HardwareSpec, NoiseModel
from qiskit_ibm_runtime.fake_provider import *

# basis_gates_dict = dict()

# for hardware_spec in HardwareSpec:
#     noise_model = NoiseModel(hardware_spec, thermal_relaxation=False)
#     if noise_model.basis_gates not in basis_gates_dict.keys():
#         if noise_model.basis_gates != BasisGates.TYPE9:
#             basis_gates_dict[noise_model.basis_gates.name] = get_ibm_noise_model(hardware_spec, thermal_relaxation=False).basis_gates

# print("There are: ", len(basis_gates_dict), "basis gates types.")
# print(list(basis_gates_dict.keys()))
def get_decompositions(basis_gates):
    phi = Parameter('phi')
    lambda_ = Parameter("lambda")
    theta = Parameter("theta")
    
    print("RY GATE")
    qc = QuantumCircuit(1)
    qc.ry(phi, 0)
    # Transpile the circuit to match the backend's basis gates
    transpiled_qc = transpile(qc, basis_gates=basis_gates)
    print(transpiled_qc)
    print()
    
    print("H GATE")
    qc = QuantumCircuit(1)
    qc.h(0)
    # Transpile the circuit to match the backend's basis gates
    transpiled_qc = transpile(qc, basis_gates=basis_gates)
    print(transpiled_qc)
    print()
    
    print("U3 GATE")
    qc = QuantumCircuit(1)
    qc.u(theta, phi, lambda_, 0)
    # Transpile the circuit to match the backend's basis gates
    transpiled_qc = transpile(qc, basis_gates=basis_gates)
    print(transpiled_qc)
    print()
    
    print("RX GATE")
    qc = QuantumCircuit(1)
    qc.rx(phi, 0)
    # Transpile the circuit to match the backend's basis gates
    transpiled_qc = transpile(qc, basis_gates=basis_gates)
    print(transpiled_qc)
    print()
    
    # Define an RY(θ) gate
    ry_gate = RYGate(theta)

    # Create the controlled RY gate
    cry_gate = ry_gate.control(1)  # 1 control qubit

    # Create a quantum circuit and add the controlled RY
    qc = QuantumCircuit(2)
    qc.append(cry_gate, [0, 1])  # Control on qubit 0, target on qubit 1
    transpiled_qc = transpile(qc, basis_gates=basis_gates)
    print(transpiled_qc)
    print()
        
if __name__ == "__main__":
    gates = []
    
    # for (basis_gate_type, basis_gates) in basis_gates_dict.items():
    #     print(f"********{basis_gate_type}********")
    #     get_decompositions(basis_gates)
    
    # ch
    # print("CH")
    # for (basis_gate_type, basis_gates) in basis_gates_dict.items():
    #     if basis_gate_type != BasisGates.TYPE9:
    #         print(f"********{basis_gate_type}********")
    #         qc = QuantumCircuit(2)
    #         phi = Parameter('phi')
    #         qc.ch(0, 1)
    #         transpiled_qc = transpile(qc, basis_gates=basis_gates)
    #         print(transpiled_qc)
    #         print()
    
    print("CSX")
    gate = CSXGate()
    for g in gate.decompositions:
        print(g)
    # for (basis_gate_type, basis_gates) in basis_gates_dict.items():
    #     if basis_gate_type != BasisGates.TYPE9:
    #         print(f"********{basis_gate_type}********")
    #         GateCX
    #         # /transpiled_qc = transpile(qc, basis_gates=basis_gates)
    #         print(print(GATE))
    #         print()