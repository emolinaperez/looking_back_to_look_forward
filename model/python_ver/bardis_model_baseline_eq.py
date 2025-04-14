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
params = {
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

# Time sequence (0 to 200 with step 0.01)
t = np.arange(0, 200.01, 0.01)

# # Round the time sequence to 2 decimal places
# t = np.round(t, 2)

# Solve the system of differential equations using odeint
solution = odeint(bm.run_bardis_model, state0, t, args=(params,))

# # Round the solution to 2 decimal places
# solution = np.round(solution, 2)

# Convert the solution to a DataFrame
df = pd.DataFrame(solution, columns=["Resources", "Economy", "Bureaucracy", "Pollution"])
df["time"] = t
df = df[["time", "Resources", "Economy", "Bureaucracy", "Pollution"]]

# Output CSV file path (update the path as needed)
output_path =  os.path.join(tableu_dir_path, "baseline.csv")
df.to_csv(output_path, index=False)

print(f"Simulation complete. Data written to: {output_path}")