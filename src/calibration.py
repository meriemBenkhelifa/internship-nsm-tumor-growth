"""
calibration.py

Calibration / parameter identification for the uncontrolled stochastic
NSM model, for a single mouse (Milestone 1). Simplified version:
treats the NSM model as deterministic (sigma=0) and only accounts for
measurement noise in the likelihood.

Measurement noise model: additive, constant variance S (Eq. 20 of the
paper), i.e. y = V_true + eps, eps ~ N(0, meas_sigma^2), matching how
the synthetic data is now generated in nsm_model.py.
"""

import numpy as np
from nsm_model import simulate_one_mouse


def loglike(params, observed_days, observed_volumes, meas_sigma=5.0):
    """
    Log-likelihood of the observed (noisy) volumes given NSM parameters,
    simplified: the forward simulation is deterministic (sigma=0), and
    measurement noise is additive Gaussian with constant variance
    (matching Eq. 20 of the paper).
    """
    a, b, alpha, V0 = params
    duration = max(observed_days)

    # simulate ONE deterministic trajectory (sigma=0 for this first version)
    t_grid, V_true = simulate_one_mouse(a, b, alpha, beta=1.0, sigma=0.0,
                                          V0=V0, duration=duration, dt=0.01)

    # get predicted volume at each observed day
    idx = np.searchsorted(t_grid, observed_days)
    predicted = V_true[idx]

    # additive Gaussian measurement noise, constant variance
    residuals = observed_volumes - predicted
    return -0.5 * np.sum((residuals / meas_sigma) ** 2)


def log_prior(params):
    """
    Uniform (log-space-friendly) prior on (a, b, alpha, V0).
    Bounds chosen to comfortably contain the true simulation parameters.
    """
    a, b, alpha, V0 = params
    if not (0.1 < a < 5.0):
        return -np.inf
    if not (0.01 < b < 1.0):
        return -np.inf
    if not (0.3 < alpha < 0.99):
        return -np.inf
    if not (5.0 < V0 < 200.0):
        return -np.inf
    return 0.0


def log_posterior(params, observed_days, observed_volumes):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglike(params, observed_days, observed_volumes)
