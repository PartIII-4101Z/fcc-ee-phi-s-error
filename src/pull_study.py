#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Pseudo-experiment driver for the B_s -> J/psi phi toy.

Runs N_TOYS pseudo-experiments at a given (omega, f_b) point,
fits each one, and returns the per-toy fitted uncertainties and
pulls. Used by the top-level scripts to perform mistag and
background-fraction scans.
"""

import numpy as np
from iminuit import Minuit

from src.config import (
    C_TRUE, S_TRUE, PHI_TRUE, D_TRUE,
    N_FULL, N_TOYS,
    EFF_SIG, EFF_BKG,
)
from src.event_generation import (
    generate_decay_times, generate_signal_mass, generate_background_mass
)
from src.likelihood import build_nll

def run_single_toy(omega, f_b, n_sig_fixed=80_000):
    """
    Generate and fit one pseudo-experiment at (omega, f_b).

    Parameters
    ----------
    omega : float
        Mistag rate applied to signal events before the fit sees them.
    f_b : float
        True background fraction used to set the number of background events.
    n_sig_fixed : int
        Number of signal events to generate per toy. Held fixed across
        f_b so that the fit is not effectively fitting the event yield.

    Returns
    -------
    dict or None
        Dictionary of fit outputs (sigma_S, sigma_phi, pull_S, pull_phi,
        f_actual, weight), or None if the fit did not converge.
    """
    # -----------------------------------------------------------
    # Step 1: decide event yields and split signal into B/Bbar
    # -----------------------------------------------------------
    n_bkg = int(f_b / (1 - f_b) * n_sig_fixed)
    n_B = np.random.binomial(n_sig_fixed, 0.5)
    n_Bbar = n_sig_fixed - n_B

    # -----------------------------------------------------------
    # Step 2: generate decay times for each category
    # -----------------------------------------------------------
    t_B = generate_decay_times(+1, C_TRUE, S_TRUE, D_TRUE, n_B)
    t_Bbar = generate_decay_times(-1, C_TRUE, S_TRUE, D_TRUE, n_Bbar)
    t_bkg = generate_decay_times(0, C_TRUE, S_TRUE, D_TRUE, n_bkg)

    # -----------------------------------------------------------
    # Step 3: generate matching masses for each category
    # -----------------------------------------------------------
    m_sig_B = generate_signal_mass(len(t_B))
    m_sig_Bbar = generate_signal_mass(len(t_Bbar))
    m_bkg = generate_background_mass(len(t_bkg))

    # -----------------------------------------------------------
    # Step 4: apply (currently trivial) signal/background efficiencies
    # -----------------------------------------------------------
    keep_B = np.random.rand(len(t_B)) < EFF_SIG
    keep_Bbar = np.random.rand(len(t_Bbar)) < EFF_SIG
    keep_bkg = np.random.rand(len(t_bkg)) < EFF_BKG

    t_B, m_sig_B = t_B[keep_B], m_sig_B[keep_B]
    t_Bbar, m_sig_Bbar = t_Bbar[keep_Bbar], m_sig_Bbar[keep_Bbar]
    t_bkg, m_bkg = t_bkg[keep_bkg], m_bkg[keep_bkg]

    # -----------------------------------------------------------
    # Step 5: concatenate everything and assign tag values
    # -----------------------------------------------------------
    m_all = np.concatenate([m_sig_B, m_sig_Bbar, m_bkg])
    t_all = np.concatenate([t_B, t_Bbar, t_bkg])
    q_all = np.concatenate([
        np.ones(len(t_B)),
        -np.ones(len(t_Bbar)),
        np.zeros(len(t_bkg)),
    ])

    # -----------------------------------------------------------
    # Step 6: re-derive the observed background fraction
    # (it differs from f_b after acceptance-rejection thinning)
    # and compute the weight used to rescale uncertainties.
    # -----------------------------------------------------------
    n_sig_obs = len(t_B) + len(t_Bbar)
    n_bkg_obs = len(t_bkg)
    n_actual = n_sig_obs + n_bkg_obs
    f_actual = n_bkg_obs / n_actual
    weight = N_FULL / n_actual

    # -----------------------------------------------------------
    # Step 7: apply mistag — flip a fraction omega of signal tags
    # -----------------------------------------------------------
    n_sig = len(t_B) + len(t_Bbar)
    flip = np.random.rand(n_sig) < omega
    q_all[:n_sig][flip] *= -1

    # -----------------------------------------------------------
    # Step 8: build and minimise the NLL
    # -----------------------------------------------------------
    nll = build_nll(t_all, m_all, q_all, omega)

    minuit = Minuit(nll, C=0.0, S=0.0, f=f_actual)
    minuit.limits["C"] = (-1, 1)
    minuit.limits["S"] = (-1, 1)
    minuit.limits["f"] = (0, 1)

    minuit.migrad()
    minuit.hesse()

    if not minuit.valid or not minuit.accurate:
        return None

    # -----------------------------------------------------------
    # Step 9: extract fit results, compute phi_s and its pull
    # -----------------------------------------------------------
    phi_fit = np.arcsin(minuit.values["S"])

    # Raw HESSE error from the 100k-event toy fit (NOT rescaled).
    # Used for the PULL, so numerator and denominator match in dataset size.
    raw_sigma_S = minuit.errors["S"]
    raw_sigma_phi = raw_sigma_S / np.sqrt(1 - S_TRUE**2)

    # Rescaled uncertainty, projected to the full FCC-ee dataset.
    # Used only for the REPORTED sensitivity sigma(phi_s), not the pull.
    sigma_S = raw_sigma_S / np.sqrt(weight)
    sigma_phi = sigma_S / np.sqrt(1 - S_TRUE**2)

    # Pulls use the RAW (un-rescaled) error so the pull width ~ 1.
    pull_S = (minuit.values["S"] - S_TRUE) / raw_sigma_S
    pull_phi = (phi_fit - PHI_TRUE) / raw_sigma_phi

    return {
        "sigma_S": sigma_S,            # rescaled, for reported sensitivity
        "sigma_phi": sigma_phi,        # rescaled, for reported sensitivity
        "pull_S": pull_S,              # raw-error pull, width ~ 1
        "pull_phi": pull_phi,          # raw-error pull, width ~ 1
        "f_actual": f_actual,
        "weight": weight,
    }

def run_toys_at_point(omega, f_b, n_toys=N_TOYS, verbose=True):
    """
    Run n_toys pseudo-experiments at one (omega, f_b) point and
    aggregate the per-toy results.

    Returns
    -------
    dict
        Aggregated outputs including the per-toy pull arrays, the
        mean sigma_S, the error on the mean, and the convergence count.
    """
    if verbose:
        print(f"\nRunning mistag rate omega = {omega:.2f}, f_b = {f_b:.2f}")

    sigma_S_values = []
    pull_S_values = []
    pull_phi_values = []
    f_actual_values = []

    for _ in range(n_toys):
        result = run_single_toy(omega, f_b)
        if result is None:
            continue
        sigma_S_values.append(result["sigma_S"])
        pull_S_values.append(result["pull_S"])
        pull_phi_values.append(result["pull_phi"])
        f_actual_values.append(result["f_actual"])

    n_converged = len(sigma_S_values)
    if verbose:
        print(f"  Converged: {n_converged}/{n_toys}")

    mean_sigma_S = np.mean(sigma_S_values)
    err_on_mean_sigma_S = np.std(sigma_S_values, ddof=1) / np.sqrt(n_converged)

    # Convert sigma_S -> sigma_phi via the small-S_f approximation.
    mean_sigma_phi = mean_sigma_S / np.sqrt(1 - S_TRUE**2)
    err_on_mean_sigma_phi = err_on_mean_sigma_S / np.sqrt(1 - S_TRUE**2)

    pull_S_arr = np.array(pull_S_values)
    pull_phi_arr = np.array(pull_phi_values)

    return {
        "n_converged": n_converged,
        "mean_sigma_S": mean_sigma_S,
        "err_on_mean_sigma_S": err_on_mean_sigma_S,
        "mean_sigma_phi_mrad": 1000 * mean_sigma_phi,
        "err_on_mean_sigma_phi_mrad": 1000 * err_on_mean_sigma_phi,
        "pull_S": pull_S_arr,
        "pull_phi": pull_phi_arr,
        "pull_S_mean": np.mean(pull_S_arr),
        "pull_S_width": np.std(pull_S_arr, ddof=1),
        "pull_phi_mean": np.mean(pull_phi_arr),
        "pull_phi_width": np.std(pull_phi_arr, ddof=1),
        "mean_f_actual": np.mean(f_actual_values),
    }

