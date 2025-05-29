class TainterModel:
    """
    Simplified Tainter‐style collapse model.
    Stocks (state vector order):
      0. State_Inputs (SI)
      1. State_Capacity (SC)
      2. Administrative_Complexity (AC)
    """
    def __init__(self, debug: bool = False):
        self.debug = debug

    def compute_flows(self, state, params, t=None):
        SI, SC, AC = state

        # Unpack parameters
        k_SI         = params["k_SI"]          # baseline input replenishment
        ef_AC        = params["ef_AC"]         # per-unit AC boost to inputs
        alpha_sat    = params["alpha_sat"]     # saturation for diminishing returns
        k_use        = params["k_use"]         # inputs consumption rate
        eps_SI_SC    = params["eps_SI_SC"]     # efficiency inputs→capacity
        k_SC_decay   = params["k_SC_decay"]    # capacity decay rate
        k_AC_growth  = params["k_AC_growth"]   # capacity→bureaucracy growth
        k_AC_decay   = params["k_AC_decay"]    # bureaucracy decay rate
        k_coll_SI    = params.get("k_coll_SI", 0.0)
        k_coll_SC    = params.get("k_coll_SC", 0.0)

        # — Diminishing returns from AC on inputs —
        breach     = max(0.0, AC - SC)
        ac_boost   = ef_AC * AC / (1.0 + alpha_sat * breach)

        # — State Inputs flows —
        si_in  = k_SI + ac_boost
        si_out = k_use * SI + k_coll_SI * breach

        # — State Capacity flows —
        cap_real  = eps_SI_SC * SI
        cap_decay = k_SC_decay * SC + k_coll_SC * breach

        # — Administrative Complexity flows —
        ac_grow = k_AC_growth * SC
        ac_dec  = k_AC_decay  * AC

        if self.debug:
            print(f"Flows | SI:+{si_in:.3f}/-{si_out:.3f} "
                  f"SC:+{cap_real:.3f}/-{cap_decay:.3f} "
                  f"AC:+{ac_grow:.3f}/-{ac_dec:.3f} "
                  f"breach={breach:.3f}")

        return {
            "si_in": si_in, "si_out": si_out,
            "cap_real": cap_real, "cap_decay": cap_decay,
            "ac_grow": ac_grow, "ac_dec": ac_dec
        }

    def compute_derivatives(self, state, flows):
        SI, SC, AC = state

        dSI = flows["si_in"]  - flows["si_out"] if SI > 0 else 0.0
        dSC = flows["cap_real"] - flows["cap_decay"] if SC > 0 else 0.0
        dAC = flows["ac_grow"] - flows["ac_dec"] if AC > 0 else 0.0

        if self.debug:
            print(f"Derivs| dSI={dSI:.3f} dSC={dSC:.3f} dAC={dAC:.3f}")

        return [dSI, dSC, dAC]

    def run(self, state, t, params):
        flows = self.compute_flows(state, params, t)
        return self.compute_derivatives(state, flows)
