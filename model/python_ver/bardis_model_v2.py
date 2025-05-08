class BardisModel:
    """
    BardisModel class encapsulates the computation of flows and derivatives for a given state and parameters in a socio-economic model.
    Methods:
        compute_flows(state, parameters):
        compute_derivatives(state, flows):
        run_bardis_model(state, time, parameters):
    """
    
    def compute_flows(self, state, parameters):
        """
        Compute the flows for the reconceptualized state-collapse model.

        Parameters:
        state (tuple): A tuple containing the state variables:
            - State_Inputs (float): Available material, human, and institutional inputs.
            - State_Capacity (float): The ability of the state to function effectively.
            - Administrative_Complexity (float): The size and cost of managing the state.
            - Systemic_Burden (float): Accumulated costs, inefficiencies, unrest.

        parameters (dict): Dictionary with model coefficients:
            - k_input_replenishment (float): Regeneration rate of State_Inputs.
            - ef_inputs_capacity (float): Efficiency of converting inputs & complexity into capacity.
            - ef_complexity_support (float): Boost to capacity from complexity (diminishing returns).
            - k_capacity_drain (float): Rate at which burden drains state capacity.
            - k_complexity_growth (float): Growth rate of administrative structures from capacity.
            - k_complexity_decay (float): Decay of complexity due to systemic burden. #NOTE: Not sure if this is correct
            - k_burden_accumulation (float): Rate at which activity adds to burden.
            - k_burden_reduction (float): Rate of burden relief (e.g., reform, adaptation).

        Returns:
        dict: Dictionary with flows:
            - input_replenishment
            - capacity_realization
            - capacity_drain
            - complexity_growth
            - complexity_decay
            - burden_accumulation
            - burden_reduction
        """

        # Unpack state variables
        State_Inputs, State_Capacity, Administrative_Complexity, Systemic_Burden = state

        # Unpack parameters
        k_input_replenishment = parameters["k_input_replenishment"]
        ef_inputs_capacity = parameters["ef_inputs_capacity"]
        ef_complexity_support = parameters["ef_complexity_support"]
        k_capacity_drain = parameters["k_capacity_drain"]
        k_complexity_growth = parameters["k_complexity_growth"]
        k_complexity_decay = parameters["k_complexity_decay"]
        k_burden_accumulation = parameters["k_burden_accumulation"]
        k_burden_reduction = parameters["k_burden_reduction"]

        # Flows
        input_replenishment = k_input_replenishment * State_Inputs

        capacity_realization = ef_inputs_capacity * State_Inputs * (
            1 + ef_complexity_support * Administrative_Complexity
        )

        capacity_drain = k_capacity_drain * State_Capacity * Systemic_Burden

        complexity_growth = k_complexity_growth * State_Capacity
        complexity_decay = k_complexity_decay * Administrative_Complexity * Systemic_Burden

        burden_accumulation = k_burden_accumulation * capacity_realization
        burden_reduction = k_burden_reduction * Systemic_Burden #NOTE: This might need to be adjusted

        return {
            "input_replenishment": input_replenishment,
            "capacity_realization": capacity_realization,
            "capacity_drain": capacity_drain,
            "complexity_growth": complexity_growth,
            "complexity_decay": complexity_decay,
            "burden_accumulation": burden_accumulation,
            "burden_reduction": burden_reduction,
        }

    def compute_derivatives(self, state, flows):
        """
        Compute the time-derivatives of the system based on the reconceptualized collapse model.

        Parameters:
        state (tuple): Current values of the state variables:
            - State_Inputs
            - State_Capacity
            - Administrative_Complexity
            - Systemic_Burden

        flows (dict): Dictionary containing:
            - input_replenishment
            - capacity_realization
            - capacity_drain
            - complexity_growth
            - complexity_decay
            - burden_accumulation
            - burden_reduction

        Returns:
        list: Time derivatives [dInputs, dCapacity, dComplexity, dBurden]
        """

        # Unpack state
        State_Inputs, State_Capacity, Administrative_Complexity, Systemic_Burden = state

        # Unpack flows
        input_replenishment = flows["input_replenishment"]
        capacity_realization = flows["capacity_realization"]
        capacity_drain = flows["capacity_drain"]
        complexity_growth = flows["complexity_growth"]
        complexity_decay = flows["complexity_decay"]
        burden_accumulation = flows["burden_accumulation"]
        burden_reduction = flows["burden_reduction"]

        # Derivatives
        dInputs = input_replenishment - capacity_realization if State_Inputs > 0 else 0
        dCapacity = capacity_realization - capacity_drain if State_Capacity > 0 else 0
        dComplexity = complexity_growth - complexity_decay if Administrative_Complexity > 0 else 0
        dBurden = burden_accumulation - burden_reduction if Systemic_Burden > 0 else 0

        return [dInputs, dCapacity, dComplexity, dBurden]


    def run_bardis_model(self, state, time,  parameters):
        """
        Run the Bardis model to compute the derivatives based on the given state, time, and parameters.

        Args:
            state (dict): The current state of the system.
            time (float): The current time.
            parameters (dict): The parameters required for the model.

        Returns:
            dict: The computed derivatives of the state variables.
        """

        # Compute flows
        flows = self.compute_flows(state, parameters)

        # Compute derivatives
        derivatives = self.compute_derivatives(state, flows)

        return derivatives
