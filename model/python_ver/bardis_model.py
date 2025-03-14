class BardisModel:

    def unpack_parameters(self, parameters):

        # Unpack parameters
        self.k_resources = parameters["k_resources"]
        self.k_pollution = parameters["k_pollution"]
        self.ef_economy_resources_on_prod = parameters["ef_economy_resources_on_prod"]
        self.ef_bureaucracy_on_prod = parameters["ef_bureaucracy_on_prod"]
        self.k_deprec = parameters["k_deprec"]
        self.ef_pollution_on_depreciation = parameters["ef_pollution_on_depreciation"]
        self.ef_economy_on_bureaucracy = parameters["ef_economy_on_bureaucracy"]
        self.k_bureaucracy = parameters["k_bureaucracy"]
        self.k_decay_bureaucracy = parameters["k_decay_bureaucracy"]
        self.ef_pollution_on_bureaucracy = parameters["ef_pollution_on_bureaucracy"]
        self.k_pollution_decay = parameters["k_pollution_decay"]

        return None
    
    def unpack_state(self, state):

        # Unpack state variables
        self.Resources, self.Economy, self.Bureaucracy, self.Pollution = state

        return None
    
    def computer_flows(self):

        # Compute flows
        resource_inflow = self.k_resources * self.Resources
        extractive_pollution = self.k_pollution * self.Economy * self.Resources

        production = self.ef_economy_resources_on_prod * self.Resources * self.Economy + self.ef_bureaucracy_on_prod * self.Bureaucracy
        depreciation = self.k_deprec * self.Economy + self.ef_pollution_on_depreciation * self.Pollution

        bureaucracy_creation = self.ef_economy_on_bureaucracy * self.Economy + self.k_bureaucracy * self.Bureaucracy
        bureaucracy_decay = self.k_decay_bureaucracy * self.Bureaucracy + self.ef_pollution_on_bureaucracy * self.Pollution

        pollution_abatement = self.k_pollution_decay * self.Pollution

        # Store results in a dictionary
        flows = {
            "resource_inflow": resource_inflow,
            "extractive_pollution": extractive_pollution,
            "production": production,
            "depreciation": depreciation,
            "bureaucracy_creation": bureaucracy_creation,
            "bureaucracy_decay": bureaucracy_decay,
            "pollution_abatement": pollution_abatement
        }

        return flows

    def compute_derivatives(self, flows):

        # Unpack flows
        resource_inflow = flows["resource_inflow"]
        extractive_pollution = flows["extractive_pollution"]
        production = flows["production"]
        depreciation = flows["depreciation"]
        bureaucracy_creation = flows["bureaucracy_creation"]
        bureaucracy_decay = flows["bureaucracy_decay"]
        pollution_abatement = flows["pollution_abatement"]

        # Compute derivatives with conditional checks
        dResources = resource_inflow - production - extractive_pollution if self.Resources > 0 else 0
        dEconomy = production - depreciation - bureaucracy_creation if self.Economy > 0 else 0
        dBureaucracy = bureaucracy_creation - bureaucracy_decay if self.Bureaucracy > 0 else 0
        dPollution = depreciation + bureaucracy_decay + extractive_pollution - pollution_abatement if self.Pollution > 0 else 0

        return [dResources, dEconomy, dBureaucracy, dPollution]

    def run_bardis_model(self, state, time,  parameters):

        # Unpack parameters
        self.unpack_parameters(parameters)

        # Unpack state variables
        self.unpack_state(state)

        # Compute flows
        flows = self.computer_flows()

        # Compute derivatives
        derivatives = self.compute_derivatives(flows)

        return derivatives



# def bardis_model(state, time, parameters):
#     # Unpack state variables
#     Resources, Economy, Bureaucracy, Pollution = state

#     # Unpack parameters
#     k_resources = parameters["k_resources"]
#     k_pollution = parameters["k_pollution"]
#     ef_economy_resources_on_prod = parameters["ef_economy_resources_on_prod"]
#     ef_bureaucracy_on_prod = parameters["ef_bureaucracy_on_prod"]
#     k_deprec = parameters["k_deprec"]
#     ef_pollution_on_depreciation = parameters["ef_pollution_on_depreciation"]
#     ef_economy_on_bureaucracy = parameters["ef_economy_on_bureaucracy"]
#     k_bureaucracy = parameters["k_bureaucracy"]
#     k_decay_bureaucracy = parameters["k_decay_bureaucracy"]
#     ef_pollution_on_bureaucracy = parameters["ef_pollution_on_bureaucracy"]
#     k_pollution_decay = parameters["k_pollution_decay"]

#     # Compute flows
#     resource_inflow = k_resources * Resources
#     extractive_pollution = k_pollution * Economy * Resources

#     production = ef_economy_resources_on_prod * Resources * Economy + ef_bureaucracy_on_prod * Bureaucracy
#     depreciation = k_deprec * Economy + ef_pollution_on_depreciation * Pollution

#     bureaucracy_creation = ef_economy_on_bureaucracy * Economy + k_bureaucracy * Bureaucracy
#     bureaucracy_decay = k_decay_bureaucracy * Bureaucracy + ef_pollution_on_bureaucracy * Pollution

#     pollution_abatement = k_pollution_decay * Pollution

#     # Compute derivatives with conditional checks
#     dResources = resource_inflow - production - extractive_pollution if Resources > 0 else 0
#     dEconomy = production - depreciation - bureaucracy_creation if Economy > 0 else 0
#     dBureaucracy = bureaucracy_creation - bureaucracy_decay if Bureaucracy > 0 else 0
#     dPollution = depreciation + bureaucracy_decay + extractive_pollution - pollution_abatement if Pollution > 0 else 0

#     return [dResources, dEconomy, dBureaucracy, dPollution]
