import numpy as np
import pandas as pd
from scipy.integrate import odeint
import os
from tainter_model_v2 import TainterModel
from utils.utils import Utils


# Set up paths
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR_PATH = os.path.dirname(DIR_PATH)
CONFIG_DIR_PATH = os.path.join(DIR_PATH, "config")
OUTPUTS_DIR_PATH = os.path.join(MODEL_DIR_PATH, "outputs")
BASELINE_DIR_PATH = os.path.join(OUTPUTS_DIR_PATH, "baseline")

# Make sure OUTPUTS and BASELINE DIRs exist
os.makedirs(OUTPUTS_DIR_PATH, exist_ok=True)
os.makedirs(BASELINE_DIR_PATH, exist_ok=True)


# Define the dynamics model
tm = TainterModel()

# Define the Utils class for reading YAML files
utils = Utils()

# Load the configuration file
config_file_name = "baseline_config_2"
CONFIG_FILE_PATH = os.path.join(CONFIG_DIR_PATH, f"{config_file_name}.yaml")
config = utils.read_yaml(CONFIG_FILE_PATH)

# Extract the parameters from the config file
initial_state = config['initial_state']
simulation_time = config['simulation_time']
step_size = config['step_size']
model_parameters = config['model_parameters']


# Time sequence
t = np.arange(0, simulation_time + step_size, step_size)

# Define RHS wrapper
def tainters_rhs(state, time, parameters):
    return tm.run_tainters_model(state, time, parameters)


# Solve the system of differential equations using odeint
solution = odeint(tainters_rhs, initial_state, t, args=(model_parameters,))

# Convert the solution to a DataFrame
df = pd.DataFrame(solution, columns=["State_Inputs", "State_Capacity", "Administrative_Complexity", "State_Integrity"])
df["time"] = t
df = df[["time", "State_Inputs", "State_Capacity", "Administrative_Complexity", "State_Integrity"]]

# Output CSV file path
output_path =  os.path.join(BASELINE_DIR_PATH, f"{config_file_name}_output.csv")
df.to_csv(output_path, index=False)

print(f"Simulation complete. Data written to: {output_path}")