"""
calibration.py

Calibration / parameter identification for the uncontrolled stochastic
NSM model, adapting the pseudo-hierarchical Bayesian method of
Browning et al. (2024, "Predicting Radiotherapy Patient Outcomes with
Real-Time Clinical Data...") to the NSM tumor growth model
(Belkhatir et al., 2020).

Parts of this file (in particular the structure of loglike / mcmc /
get_weights) are adapted from the Julia code accompanying Browning
et al., https://github.com/ap-browning/clinical_predictions
(file analysis/inference.jl), reimplemented in Python and modified
for the NSM stochastic tumor growth model.

STATUS (Milestones 1 & 2, current):
- Single-mouse calibration: WORKING, but likelihood is simplified
  (deterministic model, sigma=0 -- process/dynamical noise is
  IGNORED). This was shown to give BIASED parameter estimates when
  the data actually contain process noise (see docs/adaptation_plan.md
  and the diagnostic figure in results/).
- Population (8-mouse) joint calibration: WORKING (shared a, b, alpha;
  individual V0 per mouse, matching Eq. 20-21 of the NSM paper). This
  is a mixed-effect joint calibration -- NOT the same as Browning et
  al.'s approach (individual calibration + KDE pooling), which has not
  been implemented yet.
- Online/sequential update (get_weights_nsm): WORKING, first version.
  Uses population samples as the "prior" and re-weights them as
  measurements arrive, following the cumulative log-likelihood
  mechanism of Browning et al.'s get_weights(). Not yet tested on a
  held-out mouse (i.e. not yet a true out-of-sample prediction test).

NEXT STEP (not yet implemented): a likelihood that properly accounts
for process/dynamical noise (sigma > 0), e.g. via an Extended Kalman
Filter as in Belkhatir et al. Section III-B, or a particle filter.
"""

import numpy as np
from scipy.integrate import solve_ivp


################################################
## SINGLE-MOUSE CALIBRATION (Milestone 1)
################################################
# Simplified likelihood: treats the NSM model as a deterministic ODE
# (process noise sigma=0) with additive, constant-variance measurement
# noise (Eq. 20 of the NSM paper). This is a first approximation --
# see module docstring above for its known limitation (biased
# estimates when the true data include process noise).

def loglike(params, observed_days, observed_volumes, meas_sigma=5.0):
    """
    Log-likelihood of the observed (noisy) volumes given NSM parameters
    (a, b, alpha, V0), for a single mouse.

    The forward model is solved as a deterministic ODE (sigma=0) with
    scipy's solve_ivp, evaluated directly at the observed days.
    """
    a, b, alpha, V0 = params
    duration = max(observed_days)

    def ode_rhs(t, V):
        return a * V**alpha - b * V

    sol = solve_ivp(ode_rhs, [0, duration], [V0],
                     t_eval=observed_days, method="RK45",
                     rtol=1e-4, atol=1e-4)

    if not sol.success:
        return -np.inf

    predicted = sol.y[0]

    # additive Gaussian measurement noise, constant variance (Eq. 20)
    residuals = observed_volumes - predicted
    return -0.5 * np.sum((residuals / meas_sigma) ** 2)


def log_prior(params):
    """
    Uniform prior on (a, b, alpha, V0). Bounds chosen to comfortably
    contain the true simulation parameters used to generate the
    synthetic data.
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


################################################
## POPULATION-LEVEL CALIBRATION (Milestone 2)
################################################
# Joint calibration on all 8 mice: a, b, alpha are shared population
# parameters (fixed effects, Eq. 20), while each mouse keeps its own
# V0 (random effect, Eq. 21). This pools information across mice,
# which the paper suggests can help resolve identifiability issues
# that appear when calibrating on a single mouse (Section III-A).
#
# NOTE: this is the NSM paper's own mixed-effect formulation, NOT
# Browning et al.'s approach (which calibrates each patient
# separately and pools posteriors afterwards via KDE -- see
# docs/adaptation_plan.md for the open question on which approach to
# prioritize going forward).

def loglike_population(params, mice_days, mice_volumes, meas_sigma=5.0):
    """
    params = [a, b, alpha, V0_1, V0_2, ..., V0_n]
    mice_days, mice_volumes: lists of arrays, one per mouse.

    Vectorized: since a, b, alpha are shared across mice, all mice are
    solved as ONE ODE system (n-dimensional state vector) instead of
    n separate solve_ivp calls -- much faster for repeated MCMC calls.
    """
    a, b, alpha = params[0], params[1], params[2]
    V0_list = np.array(params[3:])

    n_mice = len(V0_list)
    all_days = np.unique(np.concatenate(mice_days))
    all_days.sort()

    def ode_rhs(t, V):
        return a * V**alpha - b * V   # applies elementwise to the n-dim vector

    sol = solve_ivp(ode_rhs, [0, all_days[-1]], V0_list,
                     t_eval=all_days, method="RK45",
                     rtol=1e-4, atol=1e-4)

    if not sol.success:
        return -np.inf

    total_ll = 0.0
    for i in range(n_mice):
        days = mice_days[i]
        vols = mice_volumes[i]
        idx = np.searchsorted(all_days, days)
        predicted = sol.y[i][idx]
        residuals = vols - predicted
        total_ll += -0.5 * np.sum((residuals / meas_sigma) ** 2)

    return total_ll


def log_prior_population(params):
    a, b, alpha = params[0], params[1], params[2]
    V0_list = params[3:]

    if not (0.1 < a < 5.0):
        return -np.inf
    if not (0.01 < b < 1.0):
        return -np.inf
    if not (0.3 < alpha < 0.99):
        return -np.inf
    for V0 in V0_list:
        if not (5.0 < V0 < 200.0):
            return -np.inf
    return 0.0


def log_posterior_population(params, mice_days, mice_volumes):
    lp = log_prior_population(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglike_population(params, mice_days, mice_volumes)


################################################
## ONLINE / SEQUENTIAL UPDATE (Milestone 3)
################################################
# Adapted from Browning et al.'s get_weights() (analysis/inference.jl).
# Sequential/online update mechanism: for each sample of (a, b, alpha)
# from the population posterior (Milestone 2), and an associated V0,
# compute the CUMULATIVE log-likelihood as measurements arrive one at
# a time. This gives a weight per sample at each time step -- samples
# that keep fitting the incoming data stay heavily weighted, others
# get down-weighted. A weighted average of the simulated curves (using
# these weights) at any point in time gives an updated prediction
# using only the data seen so far -- this is the "real-time" mechanism
# we want to adapt from the radiotherapy paper.
#
# CURRENT LIMITATION: not yet tested on a held-out mouse (i.e. a mouse
# excluded from the population calibration in Milestone 2). The demo
# so far reuses a mouse that was part of the population fit, so it is
# not yet a true out-of-sample prediction test.

def get_weights_nsm(param_samples, V0_samples, observed_days, observed_volumes,
                     meas_sigma=5.0):
    """
    param_samples : array of shape (n_samples, 3), columns = (a, b, alpha)
    V0_samples    : array of shape (n_samples,), initial condition per sample
    observed_days, observed_volumes : the new mouse's measurements, in
        chronological order (as they "arrive" over time)

    Returns:
        weights    : array (n_samples, n_timepoints), normalized weights
                      at each time step (columns sum to 1)
        all_curves : array (n_samples, n_timepoints), simulated volume
                      trajectory of each sample at the observed days
    """
    n_samples = len(param_samples)
    n_times = len(observed_days)

    all_curves = np.full((n_samples, n_times), np.nan)
    ll_contributions = np.full((n_samples, n_times), -np.inf)

    for i, (a, b, alpha) in enumerate(param_samples):
        V0 = V0_samples[i]

        def ode_rhs(t, V):
            return a * V**alpha - b * V

        sol = solve_ivp(ode_rhs, [0, max(observed_days)], [V0],
                         t_eval=observed_days, method="RK45",
                         rtol=1e-4, atol=1e-4)

        if not sol.success:
            continue

        predicted = sol.y[0]
        all_curves[i] = predicted

        residuals = observed_volumes - predicted
        ll_contributions[i] = -0.5 * (residuals / meas_sigma) ** 2

    # cumulative log-likelihood over time = sequential/online update
    cum_ll = np.cumsum(ll_contributions, axis=1)

    # normalized weights at each time step (log-sum-exp for stability)
    weights = np.zeros_like(cum_ll)
    for t in range(n_times):
        ll_t = cum_ll[:, t]
        ll_t = ll_t - np.nanmax(ll_t)
        w = np.exp(ll_t)
        weights[:, t] = w / np.nansum(w)

    return weights, all_curves