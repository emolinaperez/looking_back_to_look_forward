import numpy as np
import pandas as pd
from scipy.stats import qmc
from tainter_model import TainterModel
import pathlib
import logging
import yaml

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
script_dir = pathlib.Path(__file__).parent.absolute()
config_dir = script_dir / "config"
root = script_dir.parent.parent
tableu_dir = root / "tableau"
logging.info(f"Root directory: {root}")

# ----------------------------
# Load configuration file
# ----------------------------
config_file = config_dir / "config.yaml"
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)
logging.info(f"Configuration loaded from {config_file}")

# Extract the parameters from the config file
sample_size = config['sample_size']
time_periods = config['time_periods']
step_size = config['step_size']


logging.info(f"Sample size: {sample_size}")
logging.info(f"Time periods: {time_periods}")
logging.info(f"Step size: {step_size}")
# ---------------------------
# Set up initial conditions
# ---------------------------

# Initial conditions for the state variables
initial_state = [1.0, 0.5, 0.1, 0.05, 0.5]

# Dynamic equilibrium parameter vector p_0
p_0 = {
    "k_input_replenishment": 0.03,
    "ef_inputs_capacity": 0.05,
    "ef_complexity_support": 0.1,
    "alpha_complexity_saturation": 0.2,
    "k_cost_complexity": 0.0,
    "k_capacity_drain": 0.02,
    "k_complexity_growth": 0.03,
    "k_complexity_decay": 0.01,
    "k_burden_accumulation": 0.04,
    "k_burden_reduction": 0.01,
    "k_burden_from_complexity": 0.02,
    "k_integrity_gain": 0.05,
    "k_integrity_loss_burden": 0.05,
    "k_integrity_loss_inputs": 0.05
}

# Time steps and integration method
t = np.arange(0, time_periods + step_size, step_size)

# # Round the time sequence to 2 decimal places
# t = np.round(t, 2)

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
    # Create a numpy array from p_0 in the same order
    p0_values = np.array([
        p_0["k_input_replenishment"],
        p_0["ef_inputs_capacity"],
        p_0["ef_complexity_support"],
        p_0["alpha_complexity_saturation"],
        p_0["k_cost_complexity"],
        p_0["k_capacity_drain"],
        p_0["k_complexity_growth"],
        p_0["k_complexity_decay"],
        p_0["k_burden_accumulation"],
        p_0["k_burden_reduction"],
        p_0["k_burden_from_complexity"],
        p_0["k_integrity_gain"],
        p_0["k_integrity_loss_burden"],
        p_0["k_integrity_loss_inputs"]
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
    try:
        sim_df = run_simulation(row, initial_state, factor_names)
        results.append(sim_df)
    except Exception as e:
        logging.warning(f"Simulation failed for run_id {row['run_id']}: {e}")

# Combine all simulation outputs
out_all = pd.concat(results, ignore_index=True)

# ---------------------------
# Write output CSV files
# ---------------------------
ensamble_path = tableu_dir / f"tainter_ensemble_python_ver_{sample_size}_{time_periods}.csv"
design_path = tableu_dir / f"exp_design_python_ver_{sample_size}_{time_periods}.csv"

out_all.to_csv(ensamble_path, index=False)
exp_df.to_csv(design_path, index=False)

logging.info(f"Output files written to {tableu_dir}")