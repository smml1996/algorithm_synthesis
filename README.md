# Hardware-Optimal Quantum Algorithms
This repository is divided into two parts:
1. The code that generates POMDPs is inside `pomdp_gen` folder.
2. The code that synthesizes algorithms is available in the folder `qalgorithm_synthesis`

The quantum hardware specifications we consider are available in Qiskit and are the following:
```python
[
    # 5-qubit backends
    FAKE_ATHENS, FAKE_BELEM, FAKE_TENERIFE,
    FAKE_LIMA, FAKE_ROME, FAKE_MANILA, 
    FAKE_SANTIAGO, FAKE_BOGOTA, FAKE_OURENSE, FAKE_YORKTOWN,
    FAKE_ESSEX, FAKE_VIGO, FAKE_BURLINGTON, FAKE_QUITO, FAKE_LONDON,

    # 7-qubit backends
    FAKE_JAKARTA, FAKE_OSLO, FAKE_PERTH, 
    FAKE_LAGOS, FAKE_NAIROBI, FAKE_CASABLANCA,

    # 14-qubit backend
    FAKE_MELBOURNE,

    # 16-qubit backend
    FAKE_GUADALUPE,

    # 20-qubit backend
    FAKE_TOKYO, FAKE_POUGHKEEPSIE, FAKE_JOHANNESBURG, FAKE_BOEBLINGEN,
    FAKE_ALMADEN, FAKE_SINGAPORE,

    # 27-qubit backend
    FAKE_MUMBAI, FAKE_PARIS, FAKE_AUCKLAND, FAKE_KOLKATA, FAKE_TORONTO, FAKE_MONTREAL, FAKE_SYDNEY, FAKE_CAIRO, FAKE_HANOI, FAKE_GENEVA,

    # 28-qubit backend
    FAKE_CAMBRIDGE,

    # 53-qubit backend
    FAKE_ROCHESTER,

    # 65-qubit backend
    FAKE_BROOKLYN, FAKE_MANHATTAN,

    # 127-qubit backend
    FAKE_WASHINGTON
]
```
## Setup
### For the POMDPs generator
Python3 is required, we used Python 3.10.13. A [virtual environment](https://docs.python.org/3/library/venv.html) is recommended. Afterwards, simply execute
```python
pip install -r requirements.txt
```
### For algorithm synthesis
First, navigate to the folder `pomdp_gen` and execute
```sh
cmake . && make
```

### Usage
#### POMDPs generator
The following code assumes that you are inside the folder `pomdp_gen`. To generate the POMDPs used in the experiments we run
```sh
python bit_flip_experiments.py [arg]
```
where `arg` is an argument that has to be one of the following:

- `genpomdp`: generates all POMDPs for selected embeddings and quantum hardware specifications using the IPMA instruction set.
- `genpomdp1`: generates all POMDPs for selected embeddings and quantum hardware specifications using the CX+H instruction set.
- `ibmtestprograms`: tests all the computed guarantees for the IPMA instruction set and compares it to the value of the corresponding simulator in Qiskit. We checked that the difference of the computed guarantee and the accuracy that the simulator yields is of order $10^{-3}$.
- `ibmtestprograms1`: tests all the computed guarantees for the CX+H instruction set and compares it to the value of the corresponding simulator in Qiskit. We checked that the difference of the computed guarantee and the accuracy that the simulator yields is of order $10^{-3}$.
- `ibmsimulatorexp`: runs all synthesized algorithms (that we determine different) for IPMA instruction set in all quantum simulators.
- `ibmsimulatorexp1`: runs all synthesized algorithms (that we determine different) for CX+H instruction set in all quantum simulators.

The analysis of synthesized algorithms is performed in the notebooks `pomdp_gen/notebooks/Results Analysis QProgram Synthesis.ipynb` and `pomdp_gen/notebooks/Results Analysis QProgram Synthesis1.ipynb` for the IPMA and CX+H instruction set respectively. You need to run this notebooks before running simulations with the aforementioned commands.

#### Algorithm synthesis
Navigate to the folder `qalgorithm_synthesis`. Given a quantum hardware, e.g. `almaden` we can synthesize the algorithms for all horizons and embeddings by running
```sh
./qprogram_synthesis almaden
```
