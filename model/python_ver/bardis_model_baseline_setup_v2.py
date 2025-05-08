import numpy as np
import pandas as pd
from scipy.integrate import odeint
import os
from bardis_model_v2 import BardisModel


# Set up paths
dir_path = os.path.dirname(os.path.realpath(__file__))
model_dir_path = os.path.join(dir_path, "..")
project_dir_path = os.path.join(model_dir_path, "..")
tableu_dir_path = os.path.join(project_dir_path, "tableau")

# Define the dynamics model
bm = BardisModel()

# Initial conditions with order State_Inputs, State_Capacity, Administrative_Complexity, Systemic_Burden

# # Low-complexity / pre-collapse configuration
# state0 = [1.0, 0.1, 0.01, 0.001] 

# # Moderate configuration
state0 = [1.0, 0.5, 0.1, 0.05]

# # High-complexity / pre-collapse configuration
# state0 = [0.8, 0.3, 0.6, 0.4]

# # Low-complexity / early-growth configuration
# state0 = [1.2, 0.2, 0.05, 0.01]


# Model parameters
baseline_parameters = {
    "k_input_replenishment": 0.03,
    "ef_inputs_capacity": 0.05,
    "ef_complexity_support": 0.1,
    "alpha_complexity_saturation": 0.2,
    "k_cost_complexity": 0.01,
    "k_capacity_drain": 0.02,
    "k_complexity_growth": 0.03,
    "k_complexity_decay": 0.01,
    "k_burden_accumulation": 0.04,
    "k_burden_reduction": 0.01,
}


# Time sequence (0 to 200 with step 0.01)
t = np.arange(0, 200.01, 0.01)

# Define RHS wrapper
def bardis_rhs(state, time, parameters):
    return bm.run_bardis_model(state, time, parameters)


# Solve the system of differential equations using odeint
solution = odeint(bardis_rhs, state0, t, args=(baseline_parameters,))

# Convert the solution to a DataFrame
df = pd.DataFrame(solution, columns=["State_Inputs", "State_Capacity", "Administrative_Complexity", "Systemic_Burden"])
df["time"] = t
df = df[["time", "State_Inputs", "State_Capacity", "Administrative_Complexity", "Systemic_Burden"]]

# Output CSV file path (update the path as needed)
output_path =  os.path.join(tableu_dir_path, "baseline.csv")
df.to_csv(output_path, index=False)

print(f"Simulation complete. Data written to: {output_path}")