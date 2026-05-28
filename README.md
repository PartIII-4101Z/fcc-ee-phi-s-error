# Time-dependent CP Violation at FCC-ee: Toy Monte Carlo Study

Monte Carlo toy framework for estimating the projected statistical precision on the CP-violating phase phi_s at FCC-ee, for the decay B_s0 -> J/psi phi.

This code accompanies the Part III project report *Time-dependent CP Violation at the Future Circular Collider* (Candidate 4101Z, University of Cambridge, Department of Physics, June 2026).

## Summary

The framework:

- Generates signal and background events via acceptance-rejection sampling from the time-dependent decay-rate PDF and a uniform background mass distribution.
- Fits an unbinned negative log-likelihood with `iminuit` to extract the CP coefficients C_f, S_f, and the background fraction f_b.
- Scans over the mistag rate omega and characterises its effect on the projected precision sigma(phi_s).
- Performs pull studies across 150 pseudo-experiments per parameter value to validate the projected uncertainty and identify systematic biases.

The main result is a projected statistical uncertainty of sigma(phi_s) = 1.20 mrad at the projected FCC-ee mistag rate of omega = 0.25 and background fraction f_b = 0.20.

## Installation

Requires Python 3.10 or later. From the project root:

```bash
pip install -r requirements.txt
```

## Reproducing the report figures

To reproduce the mistag-rate scan results:

```bash
python -m scripts.run_omega_scan
```

The script generates the pseudo-experiments, performs the fits, and writes the output figures to the project root. Average runtime for 150 toys at 16 values of omega can be expected to be ~10 minutes on most modern laptops.

## Repository structure

```
fcc-ee-phi-s-error/
├── README.md
├── requirements.txt
├── LICENSE
├── src/
│   ├── config.py              # physics constants, truth values, plot style
│   ├── event_generation.py    # acceptance-rejection sampler and mass generation
│   ├── likelihood.py          # unbinned NLL and normalisation
│   ├── pull_study.py          # pseudo-experiment driver and pull computation
│   └── plotting.py            # figure-generation utilities
└── scripts/
    └── run_omega_scan.py      # top-level entry point for the mistag scan
```

## License

Released under the MIT License — see `LICENSE`.

## Acknowledgements

Project supervised by Dr. Matt Kenzie, Cavendish Laboratory, University of Cambridge.
