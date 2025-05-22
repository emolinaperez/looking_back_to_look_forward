class TainterModel:
    """
    Tainter-style socio-economic collapse model with a Population stock and decline mechanism.
    Stocks (state vector order):
        0. State_Inputs
        1. State_Capacity
        2. Administrative_Complexity
        3. State_Integrity
        4. Population
    """
    def __init__(self, debug: bool = False):
        self.debug = debug

    # ──────────────────────────────────────────────────────────────────────────────
    # 1. FLOWS
    # ──────────────────────────────────────────────────────────────────────────────
    def compute_flows(self, state, parameters):
        State_Inputs, State_Capacity, Administrative_Complexity, State_Integrity, Population = state

        # ── Parameter unpack ───────────────────────────────────────────────────────
        k_input_replenishment    = parameters["k_input_replenishment"]
        ef_inputs_capacity       = parameters["ef_inputs_capacity"]
        ef_complexity_support    = parameters["ef_complexity_support"]
        alpha_complexity_sat     = parameters["alpha_complexity_saturation"]
        k_complexity_growth      = parameters["k_complexity_growth"]
        k_complexity_decay       = parameters.get("k_complexity_decay", 0.0)
        k_capacity_drain         = parameters.get("k_capacity_drain", 0.0)
        k_integrity_gain         = parameters.get("k_integrity_gain", 0.05)
        k_integrity_loss_inputs  = parameters.get("k_integrity_loss_inputs", 0.05)
        # Population-specific parameters
        r_pop                    = parameters.get("r_population", 0.02)
        K_pop                    = parameters.get("K_population", 1.0)
        ef_pop_inputs            = parameters.get("ef_pop_inputs", 0.0)
        ef_pop_complexity        = parameters.get("ef_pop_complexity", 0.0)
        ef_pop_capacity_drain    = parameters.get("ef_pop_capacity_drain", 0.0)
        # decline parameters
        k_pop_loss_inputs        = parameters.get("k_pop_loss_inputs", 0.01)
        k_pop_loss_integrity     = parameters.get("k_pop_loss_integrity", 0.01)

        # ── Complexity diminishing-returns boost to capacity ─────────────────────
        complexity_boost = (
            ef_complexity_support * Administrative_Complexity /
            (1 + alpha_complexity_sat * Administrative_Complexity)
        )

        # ── Base flows ───────────────────────────────────────────────────────────
        input_replenishment = k_input_replenishment * State_Inputs
        capacity_realization = ef_inputs_capacity * State_Inputs * (1 + complexity_boost)
        capacity_drain = k_capacity_drain * State_Capacity
        complexity_growth = k_complexity_growth * State_Capacity
        complexity_decay = k_complexity_decay * Administrative_Complexity
        integrity_gain = k_integrity_gain * State_Capacity
        integrity_loss = k_integrity_loss_inputs * (1 - State_Inputs)

        # ── Population-driven boosts and drains ─────────────────────────────────
        input_replenishment += ef_pop_inputs * Population
        complexity_growth   += ef_pop_complexity * Population
        capacity_drain      += ef_pop_capacity_drain * Population

        # ── Population logistic growth minus decline ────────────────────────────
        base_growth = r_pop * Population * (1 - Population / K_pop)
        # decline when resources or integrity low:
        loss_inputs = k_pop_loss_inputs * max(0, (1 - State_Inputs)) * Population
        loss_integrity = k_pop_loss_integrity * max(0, (1 - State_Integrity)) * Population
        pop_growth = base_growth - (loss_inputs + loss_integrity)

        if self.debug:
            print(
                f"Flows | in_repl {input_replenishment:.3f}  cap_real {capacity_realization:.3f}  "
                f"cap_drain {capacity_drain:.3f}  cpx_grow {complexity_growth:.3f}  "
                f"cpx_decay {complexity_decay:.3f}  integ_gain {integrity_gain:.3f}  "
                f"integ_loss {integrity_loss:.3f}  pop_grow {pop_growth:.3f}"
            )

        return {
            "input_replenishment": input_replenishment,
            "capacity_realization": capacity_realization,
            "capacity_drain": capacity_drain,
            "complexity_growth": complexity_growth,
            "complexity_decay": complexity_decay,
            "integrity_gain": integrity_gain,
            "integrity_loss": integrity_loss,
            "population_growth": pop_growth
        }

    # ──────────────────────────────────────────────────────────────────────────────
    # 2. DERIVATIVES
    # ──────────────────────────────────────────────────────────────────────────────
    def compute_derivatives(self, state, flows):
        State_Inputs, State_Capacity, Administrative_Complexity, State_Integrity, Population = state

        dInputs    = flows["input_replenishment"] - flows["capacity_realization"] if State_Inputs > 0 else 0
        dCapacity  = flows["capacity_realization"] - flows["capacity_drain"]       if State_Capacity > 0 else 0
        dComplexity= flows["complexity_growth"]   - flows["complexity_decay"]       if Administrative_Complexity > 0 else 0
        dIntegrity = flows["integrity_gain"]      - flows["integrity_loss"]        if State_Integrity > 0 else 0
        dPopulation= flows["population_growth"]

        if self.debug:
            print(
                f"Derivatives | dIn {dInputs:.3f}  dCap {dCapacity:.3f}  "
                f"dCpx {dComplexity:.3f}  dInt {dIntegrity:.3f}  dPop {dPopulation:.3f}"
            )

        return [dInputs, dCapacity, dComplexity, dIntegrity, dPopulation]

    # ──────────────────────────────────────────────────────────────────────────────
    # 3.  ODE interface
    # ──────────────────────────────────────────────────────────────────────────────
    def run_tainters_model(self, state, t, parameters):
        flows = self.compute_flows(state, parameters)
        return self.compute_derivatives(state, flows)
