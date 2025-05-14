class TainterModel:
    """
    Tainter-style socio-economic collapse model **without the Systemic_Burden stock**.
    Stocks (state vector order):
        0. State_Inputs
        1. State_Capacity
        2. Administrative_Complexity
        3. State_Integrity
    """
    def __init__(self, debug: bool = False):
        self.debug = debug

    # ──────────────────────────────────────────────────────────────────────────────
    # 1. FLOWS
    # ──────────────────────────────────────────────────────────────────────────────
    def compute_flows(self, state, parameters):
        State_Inputs, State_Capacity, Administrative_Complexity, State_Integrity = state

        # ── Parameter unpack (only those still relevant) ────────────────────────
        k_input_replenishment   = parameters["k_input_replenishment"]
        ef_inputs_capacity      = parameters["ef_inputs_capacity"]
        ef_complexity_support   = parameters["ef_complexity_support"]
        alpha_complexity_sat    = parameters["alpha_complexity_saturation"]

        k_cost_complexity       = parameters.get("k_cost_complexity", 0.0)
        k_capacity_drain        = parameters.get("k_capacity_drain", 0.0)          # now just a leak
        k_complexity_growth     = parameters["k_complexity_growth"]
        k_complexity_decay      = parameters["k_complexity_decay"]                 # now burden-independent

        k_integrity_gain        = parameters.get("k_integrity_gain", 0.05)
        k_integrity_loss_inputs = parameters.get("k_integrity_loss_inputs", 0.05)

        # ── Complexity diminishing-returns boost to capacity ────────────────────
        complexity_boost = (
            ef_complexity_support * Administrative_Complexity /
            (1 + alpha_complexity_sat * Administrative_Complexity)
        )

        # ── Flows that remain ───────────────────────────────────────────────────
        input_replenishment = k_input_replenishment * State_Inputs

        capacity_realization = ef_inputs_capacity * State_Inputs * (1 + complexity_boost)
        capacity_drain       = k_capacity_drain * State_Capacity                 # burden term gone

        complexity_growth = k_complexity_growth * State_Capacity
        complexity_decay  = k_complexity_decay  * Administrative_Complexity      # burden term gone

        integrity_gain = k_integrity_gain * State_Capacity
        integrity_loss = k_integrity_loss_inputs * (1 - State_Inputs)

        if self.debug:
            print(
                "Flows | in_repl {:.3f}  cap_real {:.3f}  cap_drain {:.3f}  "
                "cpx_grow {:.3f}  cpx_decay {:.3f}  integ_gain {:.3f}  integ_loss {:.3f}"
                .format(input_replenishment, capacity_realization, capacity_drain,
                        complexity_growth, complexity_decay, integrity_gain, integrity_loss)
            )

        return {
            "input_replenishment": input_replenishment,
            "capacity_realization": capacity_realization,
            "capacity_drain": capacity_drain,
            "complexity_growth": complexity_growth,
            "complexity_decay": complexity_decay,
            "integrity_gain": integrity_gain,
            "integrity_loss": integrity_loss
        }

    # ──────────────────────────────────────────────────────────────────────────────
    # 2. DERIVATIVES
    # ──────────────────────────────────────────────────────────────────────────────
    def compute_derivatives(self, state, flows):
        State_Inputs, State_Capacity, Administrative_Complexity, State_Integrity = state

        dInputs = flows["input_replenishment"] - flows["capacity_realization"] if State_Inputs > 0 else 0

        dCapacity = flows["capacity_realization"] - flows["capacity_drain"] if State_Capacity > 0 else 0

        dComplexity = flows["complexity_growth"] - flows["complexity_decay"] if Administrative_Complexity > 0 else 0

        dIntegrity = flows["integrity_gain"] - flows["integrity_loss"] if State_Integrity > 0 else 0

        if self.debug:
            print(
                "Derivatives | dIn {:.3f}  dCap {:.3f}  dCpx {:.3f}  dInt {:.3f}"
                .format(dInputs, dCapacity, dComplexity, dIntegrity)
            )

        return [dInputs, dCapacity, dComplexity, dIntegrity]

    # ──────────────────────────────────────────────────────────────────────────────
    # 3.  ODE interface
    # ──────────────────────────────────────────────────────────────────────────────
    def run_tainters_model(self, state, time, parameters):
        flows       = self.compute_flows(state, parameters)
        derivatives = self.compute_derivatives(state, flows)
        return derivatives