#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Event generation for the B_s -> J/psi phi toy Monte Carlo.

Implements acceptance-rejection sampling of decay times from the
time-dependent decay-rate PDF, and provides helpers for the 
Gaussian signal mass distribution and the uniform 
combinatorial-background mass distribution.

The proposal distribution is Exp(1), which exactly matches the
exponential envelope of the signal PDF. Only the modulation factor
(cosh + sinh + cos + sin) is handled by the rejection step.
"""

import numpy as np

from src.config import Y_BS, X_BS, TAU_BKG, M_BS, SIGMA_M, MASS_WINDOW

def generate_decay_times(q, C, S, D, N):
    """
    Generate decay times from the time-dependent decay-rate PDF
    via acceptance-rejection sampling.

    For background (q = 0) this returns a pure exponential with
    lifetime tau_b. For signal (q = +1 or -1) it samples from the
    full time-dependent rate including the CP-violating modulation.

    Parameters
    ----------
    q : int
        Tag flag. +1 = B_s, -1 = anti-B_s, 0 = background.
    C, S, D : float
        CP-violation coefficients (C_f, S_f, D_f) of the rate.
    N : int
        Number of *proposal* samples. The returned array is smaller
        because the acceptance-rejection step thins it.

    Returns
    -------
    np.ndarray
        Accepted decay times in units of B_s lifetimes.
    """
    if q == 0:
        return np.random.exponential(scale=TAU_BKG, size=N)

    # Draw exponential decay times (the natural envelope of the PDF).
    t = np.random.exponential(scale=1.0, size=N)

    # Construct the modulation factor (sign of C, S terms flips for B vs Bbar).
    if q == 1:
        mod = (np.cosh(Y_BS * t) + D * np.sinh(Y_BS * t)
               + C * np.cos(X_BS * t) - S * np.sin(X_BS * t))
    else:
        mod = (np.cosh(Y_BS * t) + D * np.sinh(Y_BS * t)
               - C * np.cos(X_BS * t) + S * np.sin(X_BS * t))

    # Safe global upper bound on the modulation.
    mod_max = 1 + abs(D) + abs(C) + abs(S)

    # Standard acceptance-rejection step: keep t if a uniform[0,1] draw
    # falls below mod(t) / mod_max.
    accept = np.random.rand(N) < (mod / mod_max)
    return t[accept]

def generate_signal_mass(n_events, m_bs=M_BS, sigma_m=SIGMA_M):
    """
    Draw reconstructed masses for signal events from a Gaussian
    centred on m_Bs with the FCC-ee mass resolution.
    """
    return np.random.normal(loc=m_bs, scale=sigma_m, size=n_events)


def generate_background_mass(n_events, m_bs=M_BS, half_window=MASS_WINDOW):
    """
    Draw reconstructed masses for combinatorial background events,
    uniformly distributed in [m_Bs - delta_m, m_Bs + delta_m].
    """
    return np.random.uniform(low=m_bs - half_window,
                             high=m_bs + half_window,
                             size=n_events)

