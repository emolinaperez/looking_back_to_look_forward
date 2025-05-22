import numpy as np
import pandas as pd
from scipy.stats import qmc
from tainter_model_v1 import TainterModel
import pathlib
import logging
from utils.utils import Utils

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
# Create an instance of the BardisModel class
# ---------------------------
tm = TainterModel()

# ---------------------------
# Set up file paths
# ---------------------------
SCRIPT_DIR_PATH = pathlib.Path(__file__).parent.absolute()
CONFIG_DIR_PATH = SCRIPT_DIR_PATH / "config"
MODEL_DIR_PATH = SCRIPT_DIR_PATH.parent
OUTPUTS_DIR_PATH = MODEL_DIR_PATH / "outputs"
ENSEMBLE_DIR_PATH = OUTPUTS_DIR_PATH / "ensemble"
EXP_DESIGN_DIR_PATH = OUTPUTS_DIR_PATH / "exp_design"
logging.info(f"Model directory: {MODEL_DIR_PATH}")

# ----------------------------
# Load configuration file
# ----------------------------
config_file_name = "ensemble_config_1"
CONFIG_FILE_PATH = CONFIG_DIR_PATH / f"{config_file_name}.yaml"
utils = Utils()
config = utils.read_yaml(CONFIG_FILE_PATH)
logging.info(f"Configuration loaded from {CONFIG_FILE_PATH}")

# Extract the parameters from the config file
sample_size = config['sample_size']
simulation_time = config['simulation_time']
step_size = config['step_size']
initial_state = config['initial_state']
model_parameters = config['model_parameters']

logging.info(f"Sample size: {sample_size}")
logging.info(f"Simulation time: {simulation_time}")
logging.info(f"Step size: {step_size}")
logging.info(f"Initial state: {initial_state}")

# Time steps and integration method
t = np.arange(0, simulation_time + step_size, step_size)


# Our integration method is "euler" (using our custom Euler integrator)

# ---------------------------
# Create experimental design using Latin Hypercube Sampling
# ---------------------------

# Define factor names
factor_names = [
    "k_input_replenishment",
    "ef_inputs_capacity",
    "ef_complexity_support",
    "alpha_complexity_saturation",
    "k_cost_complexity",
    "k_capacity_drain",
    "k_complexity_growth",
    "k_complexity_decay",
    "k_burden_accumulation",
    "k_burden_reduction",
    "k_burden_from_complexity",
    "k_integrity_gain",
    "k_integrity_loss_burden",
    "k_integrity_loss_inputs"
]

n_factors = len(factor_names)

# Generate LHS in [0,1] and then scale to [0.5, 1.5]
sampler = qmc.LatinHypercube(d=n_factors, seed=55555)
sample = sampler.random(n=sample_size)
scaled_sample = 0.5 + sample * (1.5 - 0.5)
exp_df = pd.DataFrame(scaled_sample, columns=factor_names)
exp_df['run_id'] = np.arange(1, sample_size + 1)

# ---------------------------
# Function to run simulation for one experimental design row
# ---------------------------
def run_simulation(row, initial_state, factor_names):
    """
    Run a simulation using the provided initial state and factor multipliers.
    Parameters:
    row (pd.Series): A pandas Series containing the multipliers for each factor.
    initial_state (list or np.array): The initial state of the system.
    factor_names (list of str): The names of the factors in the order they appear in the row.
    Returns:
    pd.DataFrame: A DataFrame containing the simulation results with columns 
                  ["Resources", "Economy", "Bureaucracy", "Pollution", "time", "run_id"].
    """
    # Extract the multipliers in the order of the factor names
    p_x = np.array([row[col] for col in factor_names])
    # Create a numpy array from initial_state in the same order
    p0_values = np.array([
        model_parameters["k_input_replenishment"],
        model_parameters["ef_inputs_capacity"],
        model_parameters["ef_complexity_support"],
        model_parameters["alpha_complexity_saturation"],
        model_parameters["k_cost_complexity"],
        model_parameters["k_capacity_drain"],
        model_parameters["k_complexity_growth"],
        model_parameters["k_complexity_decay"],
        model_parameters["k_burden_accumulation"],
        model_parameters["k_burden_reduction"],
        model_parameters["k_burden_from_complexity"],
        model_parameters["k_integrity_gain"],
        model_parameters["k_integrity_loss_burden"],
        model_parameters["k_integrity_loss_inputs"]
    ])
    
    # Compute the new parameter vector by elementwise multiplication
    new_params_values = p0_values * p_x
    
    # Map back to a dictionary with the correct parameter names for the model
    parameters = {
        "k_input_replenishment": new_params_values[0],
        "ef_inputs_capacity": new_params_values[1],
        "ef_complexity_support": new_params_values[2],
        "alpha_complexity_saturation": new_params_values[3],
        "k_cost_complexity": new_params_values[4],
        "k_capacity_drain": new_params_values[5],
        "k_complexity_growth": new_params_values[6],
        "k_complexity_decay": new_params_values[7],
        "k_burden_accumulation": new_params_values[8],
        "k_burden_reduction": new_params_values[9],
        "k_burden_from_complexity": new_params_values[10],
        "k_integrity_gain": new_params_values[11],
        "k_integrity_loss_burden": new_params_values[12],
        "k_integrity_loss_inputs": new_params_values[13]
        
    }

    # Define RHS wrapper
    def tainter_rhs(state, time, parameters):
        return tm.run_tainters_model(state, time, parameters)

    # Run the simulation using the Euler method
    sol = euler_integrate(tainter_rhs, initial_state, t, parameters)
    
    # Convert the solution to a DataFrame
    df_sol = pd.DataFrame(sol, columns=["State_Inputs", "State_Capacity", "Administrative_Complexity", "Systemic_Burden", "State_Integrity"])
    df_sol["time"] = t
    
    # Subset: keep every 0.2 time unit (with dt=0.01, every 20th step)
    df_sol = df_sol.iloc[::20, :].reset_index(drop=True)
    df_sol["run_id"] = row["run_id"]
    return df_sol

# ---------------------------
# Run simulations for each experiment
# ---------------------------
logging.info("Running simulations...")
results = []
for _, row in exp_df.iterrows():
    # try:
    sim_df = run_simulation(row, initial_state, factor_names)
    results.append(sim_df)
    # except Exception as e:
    #     logging.warning(f"Simulation failed for run_id {row['run_id']}: {e}")

# Combine all simulation outputs
out_all = pd.concat(results, ignore_index=True)

# ---------------------------
# Write output CSV files
# ---------------------------
ENSEMBLE_OUTPUT_PATH = ENSEMBLE_DIR_PATH / f"{config_file_name}_output.csv"
EXP_DESIGN_OUTPUT_PATH = EXP_DESIGN_DIR_PATH / f"exp_design_{config_file_name}_output.csv"

out_all.to_csv(ENSEMBLE_OUTPUT_PATH, index=False)
exp_df.to_csv(EXP_DESIGN_OUTPUT_PATH, index=False)

logging.info(f"Output files written to {ENSEMBLE_OUTPUT_PATH} and {EXP_DESIGN_OUTPUT_PATH}")
# ---------------------------
# End of script
# ---------------------------