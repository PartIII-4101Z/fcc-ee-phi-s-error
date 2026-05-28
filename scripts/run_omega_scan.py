#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Run the mistag-rate scan for the FCC-ee phi_s toy Monte Carlo.

For each value of omega in the configured grid, this script runs
N_TOYS pseudo-experiments, fits each one, and accumulates the
per-toy uncertainties and pulls. It then produces the three figures
that appear in Section 5 of the report:

    - sigma(phi_s) vs tagging power     (Figure 5.x)
    - sigma(phi_s) vs mistag rate omega (Figure 5.1)
    - pull distributions at selected omega (Figure 5.3)

Background fraction f_b is held fixed at 0.2 (the FCC-ee projected
value). To scan over f_b instead, see scripts/run_fb_scan.py.
"""

import numpy as np

from src.config import OMEGA_VALUES
from src.pull_study import run_toys_at_point
from src.plotting import (
    plot_sigma_vs_tagging_power,
    plot_sigma_vs_mistag,
    plot_pull_histograms,
)


def main():
    # ----------------------------------------------------------
    # Configuration for this scan
    # ----------------------------------------------------------
    f_b = 0.20

    # ----------------------------------------------------------
    # Storage for scan outputs
    # ----------------------------------------------------------
    sigma_phi_mrad = []
    sigma_phi_err_mrad = []
    pull_phi_mean = []
    pull_phi_width = []
    pull_phi_all = []
    n_converged_list = []

    # ----------------------------------------------------------
    # Run the scan
    # ----------------------------------------------------------
    print(f"\nRunning omega scan at f_b = {f_b:.2f}")
    for omega in OMEGA_VALUES:
        result = run_toys_at_point(omega, f_b)

        sigma_phi_mrad.append(result["mean_sigma_phi_mrad"])
        sigma_phi_err_mrad.append(result["err_on_mean_sigma_phi_mrad"])
        pull_phi_mean.append(result["pull_phi_mean"])
        pull_phi_width.append(result["pull_phi_width"])
        pull_phi_all.append(result["pull_phi"])
        n_converged_list.append(result["n_converged"])

    # Convert to arrays for plotting.
    sigma_phi_mrad = np.array(sigma_phi_mrad)
    sigma_phi_err_mrad = np.array(sigma_phi_err_mrad)
    pull_phi_mean = np.array(pull_phi_mean)
    pull_phi_width = np.array(pull_phi_width)
    n_converged_arr = np.array(n_converged_list)

    # Error on the pull mean = pull width / sqrt(n_converged) for each omega.
    pull_phi_err = pull_phi_width / np.sqrt(n_converged_arr)

    # ----------------------------------------------------------
    # Produce the three report figures
    # ----------------------------------------------------------
    plot_sigma_vs_tagging_power(OMEGA_VALUES, sigma_phi_mrad, sigma_phi_err_mrad,
                                pull_phi_mean, pull_phi_err)

    plot_sigma_vs_mistag(OMEGA_VALUES, sigma_phi_mrad, sigma_phi_err_mrad,
                         pull_phi_mean, pull_phi_err)

    plot_pull_histograms(OMEGA_VALUES, pull_phi_all, f_b)


if __name__ == "__main__":
    main()

