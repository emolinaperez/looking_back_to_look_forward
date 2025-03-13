import numpy as np
import pandas as pd
from scipy.integrate import odeint
import os



# Define the dynamics model (differential equations)
def dynamics_model(state, t, params):
    # Unpack the state variables
    Resources, Economy, Bureaucracy, Pollution = state

    # Unpack parameters
    k_resources = params["k_resources"]
    ef_economy_resources_on_prod = params["ef_economy_resources_on_prod"]
    ef_bureaucracy_on_prod = params["ef_bureaucracy_on_prod"]
    k_deprec = params["k_deprec"]
    ef_pollution_on_depreciation = params["ef_pollution_on_depreciation"]
    k_bureaucracy = params["k_bureaucracy"]
    ef_economy_on_bureaucracy = params["ef_economy_on_bureaucracy"]
    k_decay_bureaucracy = params["k_decay_bureaucracy"]
    ef_pollution_on_bureaucracy = params["ef_pollution_on_bureaucracy"]
    k_pollution = params["k_pollution"]
    k_pollution_decay = params["k_pollution_decay"]

    # Flows
    # Resources
    resource_inflow = k_resources * Resources
    extractive_pollution = k_pollution * Economy * Resources

    # Economy
    production = ef_economy_resources_on_prod * Resources * Economy + ef_bureaucracy_on_prod * Bureaucracy  
    depreciation = k_deprec * Economy + ef_pollution_on_depreciation * Pollution
    
    # Bureaucracy
    bureaucracy_creation = ef_economy_on_bureaucracy * Economy + k_bureaucracy * Bureaucracy
    bureaucracy_decay = k_decay_bureaucracy * Bureaucracy + ef_pollution_on_bureaucracy * Pollution

    # Pollution
    pollution_abatement = k_pollution_decay * Pollution

    # Differential equations (rate of change)
    dResources = resource_inflow - production - extractive_pollution
    dEconomy = production - depreciation - bureaucracy_creation
    dBureaucracy = bureaucracy_creation - bureaucracy_decay
    dPollution = depreciation + bureaucracy_decay + extractive_pollution - pollution_abatement

    return [dResources, dEconomy, dBureaucracy, dPollution]


# Set up paths
dir_path = os.path.dirname(os.path.realpath(__file__))
model_dir_path = os.path.join(dir_path, "..")
project_dir_path = os.path.join(model_dir_path, "..")
tableu_dir_path = os.path.join(project_dir_path, "tableau")

# Initial conditions (same as in the R script)
state0 = [1.0, 1.0, 1.0, 1.0]

# Model parameters
params = {
    "k_resources":  0.15,             # Autoregeneration rate of resources  
    "ef_economy_resources_on_prod": 0.08,   # Production rate from resources
    "ef_bureaucracy_on_prod": 0.02,    # Effect of bureaucracy on production 
    "k_deprec": 0.01,                # Depreciation rate
    "ef_pollution_on_depreciation": 0.05,   # Effect of pollution on economy depreciation 
    "k_bureaucracy": 0.01,           # Bureaucracy formation rate
    "ef_economy_on_bureaucracy": 0.03, # Effect of the economy on bureaucracy formation 
    "k_decay_bureaucracy": 0.02,     # Bureaucracy decay rate
    "ef_pollution_on_bureaucracy": 0.02, # Effect of pollution on bureaucracy decay  
    "k_pollution": 0.05,             # Pollution generation rate 
    "k_pollution_decay": 0.150       # Pollution decay rate
}

# Time sequence (0 to 200 with step 0.01)
t = np.arange(0, 200.01, 0.01)

# Round the time sequence to 2 decimal places
t = np.round(t, 2)

# Solve the system of differential equations using odeint
solution = odeint(dynamics_model, state0, t, args=(params,))

# Round the solution to 2 decimal places
solution = np.round(solution, 2)

# Convert the solution to a DataFrame
df = pd.DataFrame(solution, columns=["Resources", "Economy", "Bureaucracy", "Pollution"])
df["time"] = t
# Rearranging columns so that time comes first
df = df[["time", "Resources", "Economy", "Bureaucracy", "Pollution"]]

# Output CSV file path (update the path as needed)
output_path =  os.path.join(tableu_dir_path, "baseline_python_ver.csv")
df.to_csv(output_path, index=False)

print(f"Simulation complete. Data written to: {output_path}")