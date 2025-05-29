class TainterModel:
    """
    Revised Tainter-style socio-economic collapse model.
    Stocks (state order):
        0. Natural_Resources
        1. State_Inputs
        2. State_Capacity
        3. Bureaucracy
        4. Complexity
        5. Population
    """
    def __init__(self, debug=False):
        self.debug = debug

    def compute_flows(self, state, params, t=None):
        NR, SI, SC, B, C, P = state

        # Unpack parameters
        k_NR_regen  = params["k_NR_regen"]
        k_NR_ext    = params["k_NR_ext"]
        k_SI        = params["k_SI"]
        c_econ      = params["c_econ"]
        c_bur       = params["c_bur"]
        k_use       = params["k_use"]
        eps_SI_SC   = params["eps_SI_SC"]
        k_SC_decay  = params["k_SC_decay"]
        k_B_growth  = params["k_B_growth"]
        k_B_decay   = params["k_B_decay"]
        k_C_pop     = params["k_C_pop"]
        # k_C_bur     = params.get("k_C_bur", 0.0)  # Optional parameter 
        k_C_time    = params.get("k_C_time", 0.0)
        k_C_decay   = params.get("k_C_decay", 0.0)
        r_pop       = params["r_pop"]
        K_pop       = params["K_pop"]
        k_coll_P    = params.get("k_coll_P", 0.0)
        k_coll_SC   = params.get("k_coll_SC", 0.0)
        k_coll_SI   = params.get("k_coll_SI", 0.0)

        # Natural resources
        nr_in    = k_NR_regen * NR
        nr_out   = k_NR_ext   * SI

        # State inputs
        si_in    = k_SI * (NR + c_econ * P + c_bur * P * B)
        si_out   = k_use * SI

        # State capacity
        cap_real  = eps_SI_SC * SI
        cap_decay = k_SC_decay * SC

        # Bureaucracy
        bur_grow  = k_B_growth * SC
        bur_decay = k_B_decay * B

        # Complexity
        cpx_grow  = k_C_pop * P + k_C_time
        cpx_decay = k_C_decay * C

        # Collapse breach
        breach    = max(0.0, C - SC)
        coll_P    = k_coll_P  * breach
        coll_SC   = k_coll_SC * breach
        coll_SI   = k_coll_SI * breach

        # Population logistic growth
        pop_grow = r_pop * P * (1 - P / K_pop) #NOTE: Not sure about this one, it might not need to be logistic

        if self.debug:
            print(f"Flows | nr_in {nr_in:.3f} nr_out {nr_out:.3f} "
                  f"si_in {si_in:.3f} si_out {(si_out+coll_SI):.3f} "
                  f"cap_real {cap_real:.3f} cap_decay {(cap_decay+coll_SC):.3f} "
                  f"bur_grow {bur_grow:.3f} bur_decay {bur_decay:.3f} "
                  f"cpx_grow {cpx_grow:.3f} cpx_decay {cpx_decay:.3f} "
                  f"pop_grow {pop_grow:.3f} coll_P {coll_P:.3f}")

        return {
            "nr_in": nr_in,
            "nr_out": nr_out,
            "si_in": si_in,
            "si_out": si_out + coll_SI,
            "cap_real": cap_real,
            "cap_decay": cap_decay + coll_SC,
            "bur_grow": bur_grow,
            "bur_decay": bur_decay,
            "cpx_grow": cpx_grow,
            "cpx_decay": cpx_decay,
            "pop_grow": pop_grow,
            "pop_collapse": coll_P,
        }

    def compute_derivatives(self, state, flows):
        NR, SI, SC, B, C, P = state

        dNR = flows["nr_in"]       - flows["nr_out"] if NR > 0 else 0.0
        dSI = flows["si_in"]       - flows["si_out"] if SI > 0 else 0.0
        dSC = flows["cap_real"]    - flows["cap_decay"] if SC > 0 else 0.0
        dB  = flows["bur_grow"]    - flows["bur_decay"] if B > 0 else 0.0
        dC  = flows["cpx_grow"]    - flows["cpx_decay"] if C > 0 else 0.0
        dP  = flows["pop_grow"]    - flows["pop_collapse"] if P > 0 else 0.0

        if self.debug:
            print(f"Derivs | dNR {dNR:.3f} dSI {dSI:.3f} dSC {dSC:.3f} "
                  f"dB {dB:.3f} dC {dC:.3f} dP {dP:.3f}")

        return [dNR, dSI, dSC, dB, dC, dP]

    def run_model(self, state, t, params):
        flows = self.compute_flows(state, params, t)
        return self.compute_derivatives(state, flows)
