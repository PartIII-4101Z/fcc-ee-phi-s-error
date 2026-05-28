#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Unbinned negative log-likelihood for the B_s -> J/psi phi toy.

Implements the joint mass + decay-time PDF. The likelihood depends on three free
parameters in the default fit configuration: C_f, S_f, and f_b.

D_f is fixed to its SM value because it is essentially determined
by cos(phi_s) ~ 1 for small phi_s.
"""

import numpy as np

from src.config import Y_BS, X_BS, D_TRUE, M_BS, SIGMA_M, MASS_WINDOW, TAU_BKG


def build_nll(t_all, m_all, q_all, omega):
    """
    Build a negative-log-likelihood function for one pseudo-experiment.

    The closure captures the data arrays (t_all, m_all, q_all) and
    the mistag rate omega so that the returned function depends only
    on the three free fit parameters (C, S, f).

    Parameters
    ----------
    t_all : np.ndarray
        Decay times for all events (signal + background, all tags).
    m_all : np.ndarray
        Reconstructed masses for all events.
    q_all : np.ndarray
        Tag values: +1 for B_s, -1 for anti-B_s, 0 for background.
    omega : float
        Mistag rate applied to signal events.

    Returns
    -------
    callable
        Function with signature nll(C, S, f) suitable for iminuit.
    """
    # Pre-compute everything that does not depend on the fit parameters.
    cos_xt = np.cos(X_BS * t_all)
    sin_xt = np.sin(X_BS * t_all)
    sinh_yt = np.sinh(Y_BS * t_all)
    cosh_yt = np.cosh(Y_BS * t_all)

    # Mass PDFs (signal Gaussian, background uniform).
    gauss = ((1 / (SIGMA_M * np.sqrt(2 * np.pi)))
             * np.exp(-(m_all - M_BS)**2 / (2 * SIGMA_M**2)))
    flat = np.ones_like(m_all) / (2 * MASS_WINDOW)

    # Pre-compute tag-based masks for vectorised PDF assembly.
    mask_B = (q_all == 1)
    mask_Bbar = (q_all == -1)

    # Background decay-time PDF (pure exponential).
    bkg_time = np.exp(-t_all / TAU_BKG) / TAU_BKG

    # Trapezoidal-integration grid for the signal-PDF normalisation.
    # Need to be recomputed each call because C and S are free parameters.
    ts = np.linspace(0, 10, 2000)

    def nll(C, S, f):
        """Joint negative log-likelihood for the toy fit."""
        # Numerical normalisation of the signal decay-time PDF.
        norm_B = np.trapz(
            np.exp(-ts) * (np.cosh(Y_BS * ts) + D_TRUE * np.sinh(Y_BS * ts)
                           + C * np.cos(X_BS * ts) - S * np.sin(X_BS * ts)),
            ts
        )
        norm_Bbar = np.trapz(
            np.exp(-ts) * (np.cosh(Y_BS * ts) + D_TRUE * np.sinh(Y_BS * ts)
                           - C * np.cos(X_BS * ts) + S * np.sin(X_BS * ts)),
            ts
        )

        # Signal decay-time PDFs for each tag assumption.
        pdf_B_vals = (np.exp(-t_all)
                      * (cosh_yt + D_TRUE * sinh_yt + C * cos_xt - S * sin_xt)
                      / norm_B)
        pdf_Bbar_vals = (np.exp(-t_all)
                         * (cosh_yt + D_TRUE * sinh_yt - C * cos_xt + S * sin_xt)
                         / norm_Bbar)

        # Mix B and Bbar PDFs according to the mistag rate omega.
        signal_pdf = np.zeros_like(t_all)
        signal_pdf[mask_B] = ((1 - omega) * pdf_B_vals[mask_B]
                              + omega * pdf_Bbar_vals[mask_B])
        signal_pdf[mask_Bbar] = ((1 - omega) * pdf_Bbar_vals[mask_Bbar]
                                 + omega * pdf_B_vals[mask_Bbar])

        # Combined PDF: signal + background, weighted by f_b.
        pdf_vals = (1 - f) * signal_pdf * gauss + f * bkg_time * flat
        pdf_vals = np.clip(pdf_vals, 1e-300, None)
        return -2 * np.sum(np.log(pdf_vals))

    return nll

