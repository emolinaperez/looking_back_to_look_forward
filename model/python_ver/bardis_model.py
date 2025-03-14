class BardisModel:

    def __init__(self, parameters):
        self.parameters = parameters

    # TODO: Complete the class



def bardis_model(state, time, parameters):
    # Unpack state variables
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

    # Compute flows
    resource_inflow = k_resources * Resources
    extractive_pollution = k_pollution * Economy * Resources

    production = ef_economy_resources_on_prod * Resources * Economy + ef_bureaucracy_on_prod * Bureaucracy
    depreciation = k_deprec * Economy + ef_pollution_on_depreciation * Pollution

    bureaucracy_creation = ef_economy_on_bureaucracy * Economy + k_bureaucracy * Bureaucracy
    bureaucracy_decay = k_decay_bureaucracy * Bureaucracy + ef_pollution_on_bureaucracy * Pollution

    pollution_abatement = k_pollution_decay * Pollution

    # Compute derivatives with conditional checks
    dResources = resource_inflow - production - extractive_pollution if Resources > 0 else 0
    dEconomy = production - depreciation - bureaucracy_creation if Economy > 0 else 0
    dBureaucracy = bureaucracy_creation - bureaucracy_decay if Bureaucracy > 0 else 0
    dPollution = depreciation + bureaucracy_decay + extractive_pollution - pollution_abatement if Pollution > 0 else 0

    return [dResources, dEconomy, dBureaucracy, dPollution]
