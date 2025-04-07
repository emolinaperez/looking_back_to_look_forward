import numpy as np
import pandas as pd
from scipy.integrate import odeint
import os
from bardis_model import BardisModel


# Set up paths
dir_path = os.path.dirname(os.path.realpath(__file__))
model_dir_path = os.path.join(dir_path, "..")
project_dir_path = os.path.join(model_dir_path, "..")
tableu_dir_path = os.path.join(project_dir_path, "tableau")

# Define the dynamics model
bm = BardisModel()

# Initial conditions with order "Resources", "Economy", "Bureaucracy", "Pollution"
# state0 = [1.0, 1.0, 1.0, 1.0]
state0 = [1.0, 0.1, 0.01, 0.001] 

# Model parameters
# TODO: Update the ef_ parameters to 1 and then change all other parameters to replicate paper's figure 5
# Under figure 5 we have the parameters we need to use to replicate it
params = {
    "k_resources":  0.0,             # Autoregeneration rate of resources  
    "ef_economy_resources_on_prod": 3.4546,   # Production rate from resources
    "ef_bureaucracy_on_prod": 3.4546,    # Effect of bureaucracy on production 
    "k_deprec": 1.49,                # Depreciation rate
    "ef_pollution_on_depreciation": 1,   # Effect of pollution on economy depreciation 
    "k_bureaucracy": 18.0,           # Bureaucracy formation rate
    "ef_economy_on_bureaucracy": 1, # Effect of the economy on bureaucracy formation 
    "k_decay_bureaucracy": 49.9,     # Bureaucracy decay rate
    "ef_pollution_on_bureaucracy": 1, # Effect of pollution on bureaucracy decay  
    "k_pollution": 0.0,             # Pollution generation rate 
    "k_pollution_decay": 0.0       # Pollution decay rate
}

# Time sequence (0 to 200 with step 0.01)
t = np.arange(0, 200.01, 0.01)

# Round the time sequence to 2 decimal places
t = np.round(t, 2)

# Solve the system of differential equations using odeint
solution = odeint(bm.run_bardis_model, state0, t, args=(params,))

# Round the solution to 2 decimal places
solution = np.round(solution, 2)

# Convert the solution to a DataFrame
df = pd.DataFrame(solution, columns=["Resources", "Economy", "Bureaucracy", "Pollution"])
df["time"] = t
df = df[["time", "Resources", "Economy", "Bureaucracy", "Pollution"]]

# Compute the additional columns based on the state variables:
# "inflow" corresponds to bureaucracy_creation
df["inflow"] = params["ef_economy_on_bureaucracy"] * df["Economy"] + params["k_bureaucracy"] * df["Bureaucracy"]

# "outflow" corresponds to bureaucracy_decay
df["outflow"] = params["k_decay_bureaucracy"] * df["Bureaucracy"] + params["ef_pollution_on_bureaucracy"] * df["Pollution"]

# Output CSV file path (update the path as needed)
output_path =  os.path.join(tableu_dir_path, "baseline_python_ver.csv")
df.to_csv(output_path, index=False)

print(f"Simulation complete. Data written to: {output_path}")