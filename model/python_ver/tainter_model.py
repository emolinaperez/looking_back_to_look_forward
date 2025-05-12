class TainterModel:
    """
    TainterModel class encapsulates the computation of flows and derivatives for a given state and parameters in a socio-economic model.
    Methods:
        compute_flows(state, parameters):
        compute_derivatives(state, flows):
        run_bardis_model(state, time, parameters):
    """
    def __init__(self, debug=False):
        """
        Initialize the TainterModel class.

        Args:
            debug (bool): If True, enables debug mode for detailed logging.
        """
        self.debug = debug

    
    def compute_flows(self, state, parameters):
        """
        Compute the flows for the reconceptualized state-collapse model.

        Parameters:
        state (tuple): A tuple containing the state variables:
            - State_Inputs (float): Available material, human, and institutional inputs.
            - State_Capacity (float): The ability of the state to function effectively.
            - Administrative_Complexity (float): The size and cost of managing the state.
            - Systemic_Burden (float): Accumulated costs, inefficiencies, unrest.
            - State_Integrity (float): An abstract indicator of the overall health and functionality of the system.

        parameters (dict): Dictionary with model coefficients:
            - k_input_replenishment (float): Regeneration rate of State_Inputs.
            - ef_inputs_capacity (float): Efficiency of converting inputs & complexity into State_Capacity.
            - ef_complexity_support (float): Boost to capacity from Administrative_Complexity.
            - alpha_complexity_saturation (float): Controls the rate at which the marginal contribution of complexity saturates (i.e., diminishing returns).
            - k_cost_complexity (float): Cost of maintaining Administrative_Complexity, deducted from State_Inputs.
            - k_capacity_drain (float): Rate at which Systemic_Burden erodes State_Capacity.
            - k_complexity_growth (float): Rate at which Administrative_Complexity increases with State_Capacity.
            - k_complexity_decay (float): Rate at which Administrative_Complexity decays due to Systemic_Burden.
            - k_burden_accumulation (float): Rate at which capacity activity contributes to Systemic_Burden.
            - k_burden_reduction (float): Rate of Systemic_Burden reduction through relief mechanisms (e.g., reform, adaptation).
            - k_burden_from_complexity (float): Additional burden generated from high complexity levels (quadratic effect).
            - k_integrity_gain (float): Contribution of State_Capacity to maintaining or improving system integrity.
            - k_integrity_loss_burden (float): How much Systemic_Burden erodes State_Integrity.
            - k_integrity_loss_inputs (float): Penalty to integrity for insufficient State_Inputs.

        Returns:
        dict: Dictionary with computed flows:
            - input_replenishment
            - capacity_realization
            - capacity_drain
            - complexity_growth
            - complexity_decay
            - burden_accumulation
            - burden_reduction
            - cost_of_complexity
            - integrity_gain
            - integrity_loss
        """


        # Unpack state variables
        State_Inputs, State_Capacity, Administrative_Complexity, Systemic_Burden, State_Integrity = state

        # Unpack parameters
        k_input_replenishment = parameters["k_input_replenishment"]
        ef_inputs_capacity = parameters["ef_inputs_capacity"]
        ef_complexity_support = parameters["ef_complexity_support"]
        alpha_complexity_saturation = parameters["alpha_complexity_saturation"]  # NEW
        k_cost_complexity = parameters.get("k_cost_complexity", 0.0)  # Optional
        k_capacity_drain = parameters["k_capacity_drain"]
        k_complexity_growth = parameters["k_complexity_growth"]
        k_complexity_decay = parameters["k_complexity_decay"]
        k_burden_accumulation = parameters["k_burden_accumulation"]
        k_burden_reduction = parameters["k_burden_reduction"]
        k_integrity_gain = parameters.get("k_integrity_gain", 0.05)
        k_integrity_loss_burden = parameters.get("k_integrity_loss_burden", 0.05)
        k_integrity_loss_inputs = parameters.get("k_integrity_loss_inputs", 0.05)
        k_burden_from_complexity = parameters.get("k_burden_from_complexity", 0.0)

        # Complexity diminishing returns boost
        complexity_effective_boost = (
            ef_complexity_support * Administrative_Complexity /
            (1 + alpha_complexity_saturation * Administrative_Complexity)
        )

        # Flows
        input_replenishment = k_input_replenishment * State_Inputs
        # input_replenishment = k_input_replenishment * State_Inputs * (1 - State_Inputs)

        capacity_realization = ef_inputs_capacity * State_Inputs * (1 + complexity_effective_boost)

        capacity_drain = k_capacity_drain * State_Capacity * Systemic_Burden

        complexity_growth = k_complexity_growth * State_Capacity
        complexity_decay = k_complexity_decay * Administrative_Complexity * Systemic_Burden

        burden_accumulation = (
            k_burden_accumulation * capacity_realization +
            k_burden_from_complexity * (Administrative_Complexity**2)
        )

        burden_reduction = k_burden_reduction * Systemic_Burden

        integrity_gain = k_integrity_gain * State_Capacity
        integrity_loss = k_integrity_loss_burden * Systemic_Burden + k_integrity_loss_inputs * (1 - State_Inputs)

        # Optional: maintenance cost of complexity drains State_Inputs
        cost_of_complexity = k_cost_complexity * Administrative_Complexity

        # if self.debug True print all flows and parameters
        if self.debug:
            print("Flows:")
            print(f"input_replenishment: {input_replenishment}")
            print(f"capacity_realization: {capacity_realization}")
            print(f"capacity_drain: {capacity_drain}")
            print(f"complexity_growth: {complexity_growth}")
            print(f"complexity_decay: {complexity_decay}")
            print(f"burden_accumulation: {burden_accumulation}")
            print(f"burden_reduction: {burden_reduction}")
            print(f"cost_of_complexity: {cost_of_complexity}")
            print("Parameters:")
            print(f"k_input_replenishment: {k_input_replenishment}")
            print(f"ef_inputs_capacity: {ef_inputs_capacity}")
            print(f"ef_complexity_support: {ef_complexity_support}")
            print(f"alpha_complexity_saturation: {alpha_complexity_saturation}")
            print(f"k_cost_complexity: {k_cost_complexity}")
            print(f"k_capacity_drain: {k_capacity_drain}")
            print(f"k_complexity_growth: {k_complexity_growth}")
            print(f"k_complexity_decay: {k_complexity_decay}")
            print(f"k_burden_accumulation: {k_burden_accumulation}")
            print(f"k_burden_reduction: {k_burden_reduction}")
            print("Other:")
            print(f"complexity_effective_boost: {complexity_effective_boost}")

        return {
            "input_replenishment": input_replenishment,
            "capacity_realization": capacity_realization,
            "capacity_drain": capacity_drain,
            "complexity_growth": complexity_growth,
            "complexity_decay": complexity_decay,
            "burden_accumulation": burden_accumulation,
            "burden_reduction": burden_reduction,
            "cost_of_complexity": cost_of_complexity,
            "integrity_gain": integrity_gain,
            "integrity_loss": integrity_loss
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
            - State_Integrity

        flows (dict): Dictionary containing:
            - input_replenishment
            - capacity_realization
            - capacity_drain
            - complexity_growth
            - complexity_decay
            - burden_accumulation
            - burden_reduction
            - cost_of_complexity
            - integrity_gain
            - integrity_loss

        Returns:
        list: Time derivatives of the stocks:
            - dInputs: Net change in State_Inputs.
            - dCapacity: Net change in State_Capacity.
            - dComplexity: Net change in Administrative_Complexity.
            - dBurden: Net change in Systemic_Burden.
            - dIntegrity: Net change in State_Integrity.
        """

        # Unpack state
        State_Inputs, State_Capacity, Administrative_Complexity, Systemic_Burden, State_Integrity = state

        # Unpack flows
        input_replenishment = flows["input_replenishment"]
        capacity_realization = flows["capacity_realization"]
        capacity_drain = flows["capacity_drain"]
        complexity_growth = flows["complexity_growth"]
        complexity_decay = flows["complexity_decay"]
        burden_accumulation = flows["burden_accumulation"]
        burden_reduction = flows["burden_reduction"]
        cost_of_complexity = flows.get("cost_of_complexity", 0.0)
        integrity_gain = flows["integrity_gain"]
        integrity_loss = flows["integrity_loss"]


        # Derivatives
        dInputs = input_replenishment - capacity_realization - cost_of_complexity if State_Inputs > 0 else 0
        dCapacity = capacity_realization - capacity_drain if State_Capacity > 0 else 0
        dComplexity = complexity_growth - complexity_decay if Administrative_Complexity > 0 else 0
        dBurden = burden_accumulation - burden_reduction if Systemic_Burden > 0 else 0
        dIntegrity = integrity_gain - integrity_loss if State_Integrity > 0 else 0


        # Optional: if debug print all derivatives
        if self.debug:
            print("Derivatives:")
            print(f"dInputs: {dInputs}")
            print(f"dCapacity: {dCapacity}")
            print(f"dComplexity: {dComplexity}")
            print(f"dBurden: {dBurden}")
            print(f"dIntegrity: {dIntegrity}")

        return [dInputs, dCapacity, dComplexity, dBurden, dIntegrity]


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
