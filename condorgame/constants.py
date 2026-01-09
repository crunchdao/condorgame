"""
Constants used across the condor game package.
"""

# ------------------------------------------------------------------
# CRPS configuration
#
# CRPS is computed as:
#
#     CRPS = ∫ (F(z) − 1[z ≥ x])² dz ,  z ∈ [t_min, t_max]
#
# where F(z) is the forecast CDF and x is the realized return.
#
# - `base_step` (seconds) defines the reference forecast resolution.
#   CRPS integration bounds are scaled relative to this step so that
#   scores remain comparable across different temporal resolutions.
#
# - `t[asset]` specifies the base half-width of the CRPS integration
#   range for each asset at the reference resolution. This value
#   represents a typical maximum price move to cover most of the
#   predictive mass while keeping integration finite and stable.
#
# - `num_points` is the number of discretization points used to 
#   numerically approximate the CRPS integral. Higher values improve 
#   accuracy but increase computation time.
#
# For steps larger than `base_step`, integration bounds are expanded
# by sqrt(step / base_step) to reflect increased uncertainty over
# longer time intervals.
#
# Check `crps_integral` in tracker_evaluator.py for more information
CRPS_BOUNDS = {
    "base_step": 300,
    "t":{
        "BTC": 1500,
        "SOL": 4,
        "ETH": 80,
        "XAU": 28,
    },
    "num_points": 256
}
# ------------------------------------------------------------------