import numpy as np
import pandas as pd
from scipy.stats import qmc
from bardis_model import BardisModel
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
bm = BardisModel()

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


logging.info(f"Sample size: {sample_size}")
# ---------------------------
# Set up initial conditions
# ---------------------------

# Initial conditions for the state variables
initial_state = [1.0, 0.1, 0.01, 0.001]

# Dynamic equilibrium parameter vector p_0
p_0 = {
    'k_resources': 0.15 * 0.1 * 0.08,  # Autoregeneration rate of resources  
    'ef_economy_resources_on_prod': 17.0 * 0.08,   # Production rate
    'ef_bureaucracy_on_prod': 0.02 * 0.08,  # Effect of bureaucracy on production 
    'k_deprec': 0.001 * 0.1 * 0.08,  # Depreciation rate
    'ef_pollution_on_depreciation': 0.05 * 0.08,  # Effect of pollution on economy depreciation 
    'k_bureaucracy': 0.01 * 0.08,  # Bureaucracy formation rate
    'ef_economy_on_bureaucracy': 3.5 * 0.8 * 0.08,  # Effect of the Economy on bureaucracy formation
    'k_decay_bureaucracy': 0.5 * 5 * 0.08,  # Bureaucracy decay rate
    'ef_pollution_on_bureaucracy': 0.02 * 0.08,  # Effect of pollution on bureaucracy decay  
    'k_pollution': 0.12 * 0.08,  # Pollution generation rate 
    'k_pollution_decay': 0.0 * 0.08  # Pollution decay rate
}

# Time steps and integration method
t = np.arange(0, 200.01, 0.01)

# # Round the time sequence to 2 decimal places
# t = np.round(t, 2)

# Our integration method is "euler" (using our custom Euler integrator)

# ---------------------------
# Create experimental design using Latin Hypercube Sampling
# ---------------------------

# Define factor names
factor_names = [
    'k_resources:X', 
    'ef_economy_resources_on_prod:X', 
    'ef_bureaucracy_on_prod:X', 
    'k_deprec:X',
    'ef_pollution_on_depreciation:X',
    'k_bureaucracy:X', 
    'ef_economy_on_bureaucracy:X', 
    'k_decay_bureaucracy:X', 
    'ef_pollution_on_bureaucracy:X', 
    'k_pollution:X',
    'k_pollution_decay:X'
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
        p_0["k_resources"],
        p_0["ef_economy_resources_on_prod"],
        p_0["ef_bureaucracy_on_prod"],
        p_0["k_deprec"],
        p_0["ef_pollution_on_depreciation"],
        p_0["k_bureaucracy"],
        p_0["ef_economy_on_bureaucracy"],
        p_0["k_decay_bureaucracy"],
        p_0["ef_pollution_on_bureaucracy"],
        p_0["k_pollution"],
        p_0["k_pollution_decay"]
    ])
    
    # Compute the new parameter vector by elementwise multiplication
    new_params_values = p0_values * p_x
    
    # Map back to a dictionary with the correct parameter names for the model
    parameters = {
        "k_resources": new_params_values[0],
        "ef_economy_resources_on_prod": new_params_values[1],
        "ef_bureaucracy_on_prod": new_params_values[2],
        "k_deprec": new_params_values[3],
        "ef_pollution_on_depreciation": new_params_values[4],
        "k_bureaucracy": new_params_values[5],
        "ef_economy_on_bureaucracy": new_params_values[6],
        "k_decay_bureaucracy": new_params_values[7],
        "ef_pollution_on_bureaucracy": new_params_values[8],
        "k_pollution": new_params_values[9],
        "k_pollution_decay": new_params_values[10]
    }

    # Run the simulation using the Euler method
    sol = euler_integrate(bm.run_bardis_model, initial_state, t, parameters)
    
    # Convert the solution to a DataFrame
    df_sol = pd.DataFrame(sol, columns=["Resources", "Economy", "Bureaucracy", "Pollution"])
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
ensamble_path = tableu_dir / f"bardis_ensemble_python_ver_{sample_size}.csv"
design_path = tableu_dir / f"exp_design_python_ver_{sample_size}.csv"

out_all.to_csv(ensamble_path, index=False)
exp_df.to_csv(design_path, index=False)

logging.info(f"Output files written to {tableu_dir}")