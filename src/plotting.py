#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Plotting utilities for the FCC-ee phi_s toy Monte Carlo.

Produces three families of figures used in the report:

1. sigma(phi_s) versus tagging power P,
2. sigma(phi_s) versus mistag rate omega,
3. multi-panel pull-distribution histograms for selected omega values.

Each function takes the scan-output arrays produced by the
top-level scripts and writes a labelled, publication-style figure.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_sigma_vs_tagging_power(omega_values, sigma_phi_mrad, sigma_phi_err_mrad,
                                pull_phi_mean, pull_phi_err,
                                colour='#0072B2',
                                outfile='taggingpower_errors_FCC_phi_background_unbiased.png'):
    """
    Plot the projected sigma(phi_s) and pull mean as a function of
    tagging power P = epsilon_tag * (1 - 2 omega)^2.

    A theoretical curve sigma_0 / sqrt(P) is overlaid, normalised
    to match the first scan point.
    """
    fig, ax_left = plt.subplots(figsize=(6, 5.5))

    # Convert mistag rates to tagging powers (perfect tag efficiency assumed).
    p_values = (1 - 2 * omega_values)**2

    # Theory curve: sigma scales as 1/sqrt(P).
    sigma0 = sigma_phi_mrad[0] * np.sqrt(p_values[0])
    p_theory = np.linspace(p_values.min(), p_values.max(), 200)
    sigma_theory = np.abs(sigma0 / np.sqrt(p_theory))

    ax_left.errorbar(p_values, sigma_phi_mrad, yerr=sigma_phi_err_mrad,
                     fmt='o', markersize=3, capsize=3, color=colour,
                     label=r'$\sigma_{\phi_s}$ at FCC')
    ax_left.plot(p_theory, sigma_theory, '--', color=colour, label='Theory')

    # Pull mean on the right axis.
    ax_right = ax_left.twinx()
    ax_right.errorbar(p_values, pull_phi_mean, yerr=pull_phi_err,
                      fmt='o', color='purple', markersize=3, capsize=3,
                      label=r'$\phi_s$ pull mean')
    ax_right.axhline(0, color='blue', linestyle='--', linewidth=1)
    ax_right.set_ylim(-1.1, 1.1)
    ax_right.set_ylabel("Pull mean")

    ax_left.set_xlabel(r"Tagging power $P = \varepsilon_{\mathrm{tag}}(1-2\omega)^2$")
    ax_left.set_ylabel(r"Width of $\phi_s$ / mrad")

    lines_l, labels_l = ax_left.get_legend_handles_labels()
    lines_r, labels_r = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_l + lines_r, labels_l + labels_r, loc='upper left')

    plt.title(r"Width of $\phi_s$ vs Tagging power for 8M signal 2M background")
    fig.savefig(outfile, dpi=500, bbox_inches="tight")
    plt.show()


def plot_sigma_vs_mistag(omega_values, sigma_phi_mrad, sigma_phi_err_mrad,
                         pull_phi_mean, pull_phi_err,
                         colour='#0072B2',
                         outfile='mistag_errors_FCC_phi_background_unbiased.png'):
    """
    Plot the projected sigma(phi_s) and pull mean as a function of
    the mistag rate omega.

    A theoretical curve sigma_0 / (1 - 2 omega) is overlaid,
    normalised to match the first scan point.
    """
    fig, ax_left = plt.subplots(figsize=(6, 5.5))

    sigma0 = sigma_phi_mrad[0] * (1 - 2 * omega_values[0])
    omega_theory = np.linspace(omega_values.min(), omega_values.max(), 200)
    sigma_theory = np.abs(sigma0 / (1 - 2 * omega_theory))

    ax_left.errorbar(omega_values, sigma_phi_mrad, yerr=sigma_phi_err_mrad,
                     fmt='o', markersize=3, capsize=3, color=colour,
                     label=r'$\sigma_{\phi_s}$ at FCC')
    ax_left.plot(omega_theory, sigma_theory, '--', color=colour, label='Theory')

    ax_right = ax_left.twinx()
    ax_right.errorbar(omega_values, pull_phi_mean, yerr=pull_phi_err,
                      fmt='o', color='purple', markersize=3, capsize=3,
                      label=r'$\phi_s$ pull mean')
    ax_right.axhline(0, color='blue', linestyle='--', linewidth=1)
    ax_right.set_ylim(-1.1, 1.1)
    ax_right.set_ylabel("Pull mean")

    ax_left.set_xlabel(r"Mistag rate $\omega$")
    ax_left.set_ylabel(r"Width of $\phi_s$ / mrad")

    lines_l, labels_l = ax_left.get_legend_handles_labels()
    lines_r, labels_r = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_l + lines_r, labels_l + labels_r, loc='upper left')

    plt.title(r"Width of $\phi_s$ vs Mistag rate for 8M signal 2M background")
    fig.savefig(outfile, dpi=500, bbox_inches="tight")
    plt.show()


def plot_pull_histograms(omega_values, pull_phi_all, f_b,
                         selected_omegas=(0.10, 0.20, 0.30, 0.40),
                         outfile='pull_histograms_grid.png'):
    """
    Plot a 2x2 grid of pull-distribution histograms at selected
    mistag rates. Each panel labels the corresponding tagging power
    and the sample mean and width of the pulls.

    Parameters
    ----------
    omega_values : np.ndarray
        Full array of omega values that were scanned over.
    pull_phi_all : list of np.ndarray
        For each omega, the array of per-toy phi_s pulls.
    f_b : float
        Background fraction used for the scan (printed in the figure title).
    selected_omegas : tuple of float
        The four omega values to actually plot (must be in omega_values).
    """
    # Look up the pulls corresponding to each selected omega.
    selected_pulls = []
    for w in selected_omegas:
        idx = np.argmin(np.abs(omega_values - w))
        selected_pulls.append(pull_phi_all[idx])

    selected_p = [(1 - 2 * w)**2 for w in selected_omegas]

    fig_grid, axes = plt.subplots(2, 2, figsize=(7, 6),
                                  sharex=True, sharey=True)

    for i, (p_val, pulls) in enumerate(zip(selected_p, selected_pulls)):
        ax = axes.flat[i]
        ax.hist(pulls, bins=15, density=True,
                color='#0072B2', edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', ls='--', lw=1)
        ax.axvline(np.mean(pulls), color='black', ls='-', lw=1)
        ax.set_title(rf'$P$={p_val:.2f}  '
                     rf'$\mu$={np.mean(pulls):.2f}  '
                     rf'$\sigma$={np.std(pulls, ddof=1):.2f}')

    for ax in axes[-1, :]:
        ax.set_xlabel(r'Pull of $\phi_s$')
    for ax in axes[:, 0]:
        ax.set_ylabel('Density')

    fig_grid.suptitle(rf'$\phi_s$ pull distributions across $P$ scan ($f_b$ = {f_b})')
    fig_grid.tight_layout()
    fig_grid.savefig(outfile, dpi=500, bbox_inches="tight")
    plt.show()

