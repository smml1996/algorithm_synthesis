from qiskit import QuantumCircuit
from qiskit.circuit.library import ECRGate, RZXGate, RZZGate
import numpy as np

print("ECR DECOMPOSITION")
ecr_gate = ECRGate()
for decomposition in ecr_gate.decompositions:
    print(decomposition)
 
print("*****")    
print("RZX(pi/4) gate decompsoiton")
angle = np.pi / 4
rzx_gate = RZXGate(angle)
for decomposition in rzx_gate.decompositions:
    print(decomposition)

print("*****")    
print("RZX(-pi/4) gate decompsoiton")
angle = -np.pi / 4
rzx_gate = RZXGate(angle)
for decomposition in rzx_gate.decompositions:
    print(decomposition)