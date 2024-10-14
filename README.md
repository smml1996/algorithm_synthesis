# Hardware-Optimal Quantum Algorithms<a name="hardware-optimal-quantum-algorithms"></a>
In order to synthesize algorithms we have to first generate the Partially Observable Markov Decision Process (POMDP), and then synthesize the optimal algorithm.

- [Hardware-Optimal Quantum Algorithms](#hardware-optimal-quantum-algorithms)
  - [Hardware Specifications](#hardware-specifications)
  - [Setup](#setup)
    - [For the POMDPs generator](#for-the-pomdps-generator)
    - [For algorithm synthesis](#for-algorithm-synthesis)
  - [Usage](#usage)
    - [Generate a configuration file](#generate-a-configuration-file)
    - [Generate an embeddings file](#generate-an-embeddings-file)
    - [Generate the POMDPs](#generate-the-pomdps)
    - [Synthesize algorithms](#synthesize-algorithms)
   
## Hardware Specifications<a name="hardware-specifications"></a>
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
## Setup<a name="setup"></a>
First specify the absolute path of the project root directory in the file `.settings`.

### For the POMDPs generator<a name="for-the-pomdps-generator"></a>
Python3 is required, we used Python 3.10.13. A [virtual environment](https://docs.python.org/3/library/venv.html) is recommended. Afterwards, simply execute
```python
pip install -r requirements.txt
```

### For algorithm synthesis<a name="for-algorithm-synthesis"></a>
The code that synthesizes algorithms is in C++. Compile it by navigating to the folder `qalgorithm_synthesis/` and running the following command
```sh
cmake . && make
```

## Usage<a name="usage"></a>
The steps to synthesize an algorithm can be divided into the following steps:

### Generate a configuration file<a name="generate-a-configuration-file"></a>
Configuration files are json files that might look something like:
```json
{
    "name": "B53",
    "experiment_id": "ipma",
    "min_horizon": 4,
    "max_horizon": 7,
    "output_dir": "algorithm_synthesis/results/bitflip/ipma/B53",
    "algorithms_file": "",
    "hardware": [
        "fake_rochester"
    ]
}
```
You can generate the configuration files for the paper by navigating to the folder `problems/` and running

```sh
python bitflip.py gen_configs
```
It will generate the configuration files in the directory `configs/bitflip/`.

### Generate an embeddings file<a name="generate-an-embeddings-file"></a>
It contains various mappings from virtual qubits to physical qubits. This is a json file that should be located in the specified `output_dir` of the configuration file. 

You can generate the all the embeddings files used in the paper
```sh
python bitflip.py embeddings
```
### Generate the POMDPs<a name="generate-the-pomdps"></a>
You can generate the all the POMDPs used in the paper
```sh
python bitflip.py all_pomdps
```
This will generate the corresponding pomdps for each embedding in the folder `{output_folder}/pomdps/` where `output_folder` is the path specified in the configuration file.

### Synthesize algorithms<a name="synthesize-algorithms"></a>
Navigate to the folder `qalgorithm_synthesis` and run the following command passing as argument a path to a configuration file.
```sh
./qprogram_synthesis bellmaneq ../configs/config_file.json
```
This will outputs json files containing an algorithm. The algorithm corresponding to each embedding are going to be located in `{output_folder}/algorithms/` where `output_folder` is the output folder specified in the configuration file `config_file.json`.