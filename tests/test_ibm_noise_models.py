import os, sys
sys.path.append(os.getcwd()+"/..")
from ibm_noise_models import HardwareSpec, NoiseModel


for hardware_spec in HardwareSpec:
    print(hardware_spec)
    noise_model = NoiseModel(hardware_spec, thermal_relaxation=False)