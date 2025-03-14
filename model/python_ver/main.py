import numpy as np
import pandas as pd
from scipy.stats import qmc

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
# Define the bardis_model function
# (Translated from your previous R function)
# ---------------------------
def bardis_model(state, time, parameters):
    Resources, Economy, Bureaucracy, Pollution = state

    # Unpack parameters
    k_resources = parameters["k_resources"]
    k_pollution = parameters["k_pollution"]
    ef_economy_resources_on_prod = parameters["ef_economy_resources_on_prod"]
    ef_bureaucracy_on_prod = parameters["ef_bureaucracy_on_prod"]
    k_deprec = parameters["k_deprec"]
    ef_pollution_on_depreciation = parameters["ef_pollution_on_depreciation"]
    ef_economy_on_bureaucracy = parameters["ef_economy_on_bureaucracy"]
    k_bureaucracy = parameters["k_bureaucracy"]
    k_decay_bureaucracy = parameters["k_decay_bureaucracy"]
    ef_pollution_on_bureaucracy = parameters["ef_pollution_on_bureaucracy"]
    k_pollution_decay = parameters["k_pollution_decay"]

    # Flows
    resource_inflow = k_resources * Resources
    extractive_pollution = k_pollution * Economy * Resources
    production = ef_economy_resources_on_prod * Resources * Economy + ef_bureaucracy_on_prod * Bureaucracy
    depreciation = k_deprec * Economy + ef_pollution_on_depreciation * Pollution
    bureaucracy_creation = ef_economy_on_bureaucracy * Economy + k_bureaucracy * Bureaucracy
    bureaucracy_decay = k_decay_bureaucracy * Bureaucracy + ef_pollution_on_bureaucracy * Pollution
    pollution_abatement = k_pollution_decay * Pollution

    # Rate of change with conditional checks (if state > 0, compute derivative; else 0)
    dResources = resource_inflow - production - extractive_pollution if Resources > 0 else 0
    dEconomy = production - depreciation - bureaucracy_creation if Economy > 0 else 0
    dBureaucracy = bureaucracy_creation - bureaucracy_decay if Bureaucracy > 0 else 0
    dPollution = (depreciation + bureaucracy_decay + extractive_pollution - pollution_abatement) if Pollution > 0 else 0

    return [dResources, dEconomy, dBureaucracy, dPollution]

# ---------------------------
# Set up file paths and initial conditions
# ---------------------------
root = "/Users/edmun/Library/CloudStorage/OneDrive-Personal/Edmundo-ITESM/3.Proyectos/63. Looking Back to Look Forward/looking_back_to_look_forward/"
# (Note: The model file is assumed to be incorporated via the function above)

# Initial conditions for the state variables
state = [1.0, 1.0, 1.0, 1.0]

# Dynamic equilibrium parameter vector p_0
p_0 = {
    "k_resources": 0.15 * 0.5,                    # 0.075
    "ef_economy_resources_on_prod": 0.08 * 1.5,     # 0.12
    "ef_bureaucracy_on_prod": 0.02 * 1.5,           # 0.03
    "k_deprec": 0.01,
    "ef_pollution_on_depreciation": 0.05,
    "k_bureaucracy": 0.01,
    "ef_economy_on_bureaucracy": 0.03,
    "k_decay_bureaucracy": 0.02,
    "ef_pollution_on_bureaucracy": 0.02,
    "k_pollution": 0.05,
    "k_pollution_decay": 0.150
}

# Time steps and integration method
t = np.arange(0, 200.01, 0.01)
# Our integration method is "euler" (using our custom Euler integrator)

# ---------------------------
# Create experimental design using Latin Hypercube Sampling
# ---------------------------
sample_size = 100
n_factors = 11

# Define factor names (with ":X" suffix as in your R code)
Xs = [
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

# Generate LHS in [0,1] and then scale to [0.5, 1.5]
sampler = qmc.LatinHypercube(d=n_factors, seed=55555)
sample = sampler.random(n=sample_size)
scaled_sample = 0.5 + sample * (1.5 - 0.5)
Exp = pd.DataFrame(scaled_sample, columns=Xs)
Exp['Run.ID'] = np.arange(1, sample_size + 1)

# ---------------------------
# Function to run simulation for one experimental design row
# ---------------------------
def run_simulation(row):
    # Extract the multipliers in the order of Xs
    p_x = np.array([row[col] for col in Xs])
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
    sol = euler_integrate(bardis_model, state, t, parameters)
    # Convert the solution to a DataFrame
    df_sol = pd.DataFrame(sol, columns=["Resources", "Economy", "Bureaucracy", "Pollution"])
    df_sol["time"] = t
    # Subset: keep every 0.2 time unit (with dt=0.01, every 20th step)
    df_sol = df_sol.iloc[::20, :].reset_index(drop=True)
    df_sol["Run.ID"] = row["Run.ID"]
    return df_sol

# ---------------------------
# Run simulations for each experiment
# ---------------------------
results = []
for _, row in Exp.iterrows():
    sim_df = run_simulation(row)
    results.append(sim_df)

# Combine all simulation outputs
out_all = pd.concat(results, ignore_index=True)

# ---------------------------
# Write output CSV files
# ---------------------------
ensamble_path = root + "tableau/bardis_ensamble.csv"
design_path = root + "tableau/exp_design.csv"

out_all.to_csv(ensamble_path, index=False)
Exp.to_csv(design_path, index=False)

print(f"Simulation complete. Ensemble saved to: {ensamble_path}")
print(f"Experimental design saved to: {design_path}")
