import numpy as np
import pandas as pd
from tainter_model_v6 import TainterModel
import pathlib
import logging
from utils.utils import Utils
import os

# ---------------------------
# Set up logging
# ---------------------------
logging.basicConfig(level=logging.INFO)

# ---------------------------
# Define custom Euler integrator
# ---------------------------
def euler_integrate(func, y0, t, parameters):
    dt = t[1] - t[0]
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        dy = func(y[i-1], t[i-1], parameters)
        y[i] = y[i-1] + dt * np.array(dy)
    return y

# ---------------------------
# Create model instance
# ---------------------------
tm = TainterModel()

# ---------------------------
# Set up file paths
# ---------------------------
SCRIPT_DIR_PATH   = pathlib.Path(__file__).parent.absolute()
CONFIG_DIR_PATH   = SCRIPT_DIR_PATH / "config"
MODEL_DIR_PATH    = SCRIPT_DIR_PATH.parent
OUTPUTS_DIR_PATH  = MODEL_DIR_PATH / "outputs"
SINGLE_SIM_DIR_PATH = OUTPUTS_DIR_PATH / "single_sim"
os.makedirs(OUTPUTS_DIR_PATH, exist_ok=True)

# ----------------------------
# Load configuration file
# ----------------------------
config_file_name  = "run_sim_6"     # you can rename to e.g. "sim_config"
CONFIG_FILE_PATH  = CONFIG_DIR_PATH / f"{config_file_name}.yaml"
utils             = Utils()
config            = utils.read_yaml(CONFIG_FILE_PATH)
logging.info(f"Configuration loaded from {CONFIG_FILE_PATH}")

# Extract config settings
simulation_time   = config['simulation_time']
step_size         = config['step_size']
initial_state     = config['initial_state']
model_parameters  = config['model_parameters']

logging.info(f"Simulation time: {simulation_time}")
logging.info(f"Step size: {step_size}")
logging.info(f"Initial state: {initial_state}")

# ---------------------------
# Prepare time array
# ---------------------------
t = np.arange(0, simulation_time + step_size, step_size)

# ---------------------------
# Define RHS wrapper
# ---------------------------
def tainter_rhs(state, time, parameters):
    return tm.run(state, time, parameters)

# ---------------------------
# Run single simulation
# ---------------------------
logging.info("Running single simulation…")
sol = euler_integrate(tainter_rhs, initial_state, t, model_parameters)

# Build output DataFrame
df_sol = pd.DataFrame(
    sol,
    columns=["State_Inputs", "State_Capacity", "Administrative_Complexity"]
)
df_sol["time"] = t

# Optional: down‐sample for coarser output (every 0.2 time units)
skip = int(0.2 / step_size)
if skip > 1:
    df_sol = df_sol.iloc[::skip].reset_index(drop=True)

# ---------------------------
# Write output CSV
# ---------------------------
SIM_OUTPUT_PATH = SINGLE_SIM_DIR_PATH / f"{config_file_name}_simulation_output.csv"
df_sol.to_csv(SIM_OUTPUT_PATH, index=False)
logging.info(f"Simulation output written to {SIM_OUTPUT_PATH}")
