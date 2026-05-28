#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Physics constants, default parameter values, and matplotlib styling
for the FCC-ee phi_s toy Monte Carlo.

All numerical inputs that define the underlying physics or the
expected detector performance live here, so they can be changed in
one place without hunting through scripts.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


import shutil
import matplotlib.pyplot as plt

# Use LaTeX for text rendering only if a LaTeX installation is found on
# the system PATH. This keeps figure styling identical on the development
# machine while allowing the code to run anywhere (LaTeX absent -> matplotlib's
# built-in mathtext renderer is used instead).
use_latex = shutil.which("latex") is not None

plt.rcParams.update({
    # Text rendering
    'text.usetex': use_latex,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],

    # Font sizes — match the LaTeX report typography
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,

    # Tick marks inward (standard in HEP)
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,

    # Figure size in inches
    'figure.figsize': (6, 4),
    'figure.dpi': 150,

    # Line widths
    'lines.linewidth': 1.5,
    'axes.linewidth': 0.8,
})


# ---------------------------------------------------------------
# Physics constants (B_s system)
# ---------------------------------------------------------------
# y = Delta Gamma_s / (2 Gamma_s)
# x = Delta m_s   / Gamma_s
Y_BS = 0.046
X_BS = 25.194

# Background lifetime in units of B_s lifetimes
TAU_BKG = 0.1

# B_s mass in GeV and reconstructed mass resolution
M_BS = 5.366
SIGMA_M = 0.005      # 5 MeV resolution
MASS_WINDOW = 0.1    # +/- 100 MeV mass window


# ---------------------------------------------------------------
# Truth values used to generate pseudo-data
# ---------------------------------------------------------------
C_TRUE = 0.0
S_TRUE = -0.04
PHI_TRUE = np.arcsin(S_TRUE)
D_TRUE = 0.999


# ---------------------------------------------------------------
# Toy / sample sizes
# ---------------------------------------------------------------
N_TOYS = 150            # pseudo-experiments per parameter point
N_FULL = 10_000_000     # projected FCC-ee yield (8M signal + 2M bkg)
N_EFFECTIVE = 100_000   # per-toy event count, rescaled to N_FULL via 'weight'


# ---------------------------------------------------------------
# Detector efficiencies (perfect by assumption in this study)
# ---------------------------------------------------------------
EFF_SIG = 1.0
EFF_BKG = 1.0


# ---------------------------------------------------------------
# Scan grids
# ---------------------------------------------------------------
OMEGA_VALUES = np.linspace(0.1, 0.4, 16)


# ---------------------------------------------------------------
# Useful derived quantity
# ---------------------------------------------------------------
NORM_A = (1 + D_TRUE * Y_BS) / (1 - Y_BS**2)

