"""Calibrated model parameters for applications."""

# Bansal-Yaron long-run risk model (approximate)
BANSAL_YARON = {
    "kappa_x": 0.021,
    "sigma_x": 0.0078,
    "kappa_v": 0.013,
    "sigma_v": 0.0023,
    "mu_c": 0.0015,
    "phi_c": 1.0,
    "gamma": 10.0,
    "delta": 0.998,
    "psi": 1.5,
}

# Vasicek short-rate model
VASICEK = {
    "kappa": 0.15,
    "theta": 0.05,
    "sigma": 0.015,
}
