#!/usr/bin/env python3
"""
This script uses a set of system dynamics equations to solve for the unknown parameter values.
The model equations are:

  resource_inflow      = k_resources * Resources
  extractive_pollution = k_pollution * Economy * Resources

  production           = ef_economy_resources_on_prod * Resources * Economy + ef_bureaucracy_on_prod * Bureaucracy
  depreciation         = k_deprec * Economy + ef_pollution_on_depreciation * Pollution

  bureaucracy_creation = ef_economy_on_bureaucracy * Economy + k_bureaucracy * Bureaucracy
  bureaucracy_decay    = k_decay_bureaucracy * Bureaucracy + ef_pollution_on_bureaucracy * Pollution

  pollution_abatement  = k_pollution_decay * Pollution

And the derivatives:
  dResources   = resource_inflow - production - extractive_pollution
  dEconomy     = production - depreciation - bureaucracy_creation
  dBureaucracy = bureaucracy_creation - bureaucracy_decay
  dPollution   = depreciation + bureaucracy_decay + extractive_pollution - pollution_abatement

The initial values are:
  Resources  = 1
  Economy    = 0.1
  Bureaucracy= 0.01
  Pollution  = 0.001

The target flows are:
  production            = 0.38
  depreciation          = 0.15
  bureaucracy creation  = 0.3
  bureaucracy decay     = 0.5

All ef_ parameters are set to 1.0.
We assume that k_resources, k_pollution, and k_pollution_decay are zero.

This script solves for:
  k_deprec, k_bureaucracy, k_decay_bureaucracy
"""

def main():
    # --- Initial Stocks ---
    Resources = 1.0
    Economy = 0.1
    Bureaucracy = 0.01
    Pollution = 0.001

    # --- Effect Parameters (all set to 1.0) ---
    ef_economy_resources_on_prod = 1.0
    ef_bureaucracy_on_prod = 1.0
    ef_pollution_on_depreciation = 1.0
    ef_economy_on_bureaucracy = 1.0
    ef_pollution_on_bureaucracy = 1.0

    # --- Target Flow Values ---
    production_target = 0.38
    depreciation_target = 0.15
    bureaucracy_creation_target = 0.3
    bureaucracy_decay_target = 0.5

    # --- Flows Assumed to be Zero ---
    k_resources = 0.0
    k_pollution = 0.0
    k_pollution_decay = 0.0

    # --- Compute Production ---
    computed_production = (ef_economy_resources_on_prod * Resources * Economy +
                           ef_bureaucracy_on_prod * Bureaucracy)
    print("Computed production =", computed_production)
    if abs(computed_production - production_target) > 1e-6:
        print("Warning: Computed production does not match target production of",
              production_target,
              " (difference =", production_target - computed_production, ")")
        print("You may need to adjust an ef parameter if production matching is desired.\n")

    # --- Solve for the Missing Parameters ---
    # Depreciation: depreciation = k_deprec * Economy + ef_pollution_on_depreciation * Pollution = depreciation_target
    # => k_deprec = (depreciation_target - ef_pollution_on_depreciation * Pollution) / Economy
    k_deprec = (depreciation_target - ef_pollution_on_depreciation * Pollution) / Economy

    # Bureaucracy Creation: bureaucracy_creation = ef_economy_on_bureaucracy * Economy + k_bureaucracy * Bureaucracy = bureaucracy_creation_target
    # => k_bureaucracy = (bureaucracy_creation_target - ef_economy_on_bureaucracy * Economy) / Bureaucracy
    k_bureaucracy = (bureaucracy_creation_target - ef_economy_on_bureaucracy * Economy) / Bureaucracy

    # Bureaucracy Decay: bureaucracy_decay = k_decay_bureaucracy * Bureaucracy + ef_pollution_on_bureaucracy * Pollution = bureaucracy_decay_target
    # => k_decay_bureaucracy = (bureaucracy_decay_target - ef_pollution_on_bureaucracy * Pollution) / Bureaucracy
    k_decay_bureaucracy = (bureaucracy_decay_target - ef_pollution_on_bureaucracy * Pollution) / Bureaucracy

    # --- Output the Computed Missing Parameter Values ---
    print("Missing Parameter Values:")
    print("  k_deprec           =", k_deprec)
    print("  k_bureaucracy      =", k_bureaucracy)
    print("  k_decay_bureaucracy=", k_decay_bureaucracy)

if __name__ == '__main__':
    main()
