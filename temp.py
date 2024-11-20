
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator
from ibm_noise_models import HardwareSpec, NoiseModel, get_ibm_noise_model

noise_model = get_ibm_noise_model(HardwareSpec.TENERIFE, thermal_relaxation=False)
qc = QuantumCircuit(3)

qc.h(0)
qc.cx(0,1)
qc.cx(1,2)
sim_noise = AerSimulator(method ='statevector', noise_model=noise_model)

new_circuit = transpile(qc, sim_noise, optimization_level=3, initial_layout=[5,1,2])
print(new_circuit)
for data in new_circuit.data:
    print(data.operation)
    print(data.qubits)
    print()