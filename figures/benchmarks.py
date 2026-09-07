"""Results of the one-off benchmark calculations.

These are convergence and spin-state tests rather than production results, so
they are tabulated here instead of being recomputed on every plot. The
underlying calculations are part of the calculation archive.
"""
__author__ = "Nabil Khossossi"

# Total energy against plane-wave cutoff, referred to 700 eV, in meV per atom.
# beta-Co(OH)2 at fixed geometry with all FFT grids pinned.
CUTOFF_CONVERGENCE = [(400, 0.067), (450, 3.111), (500, 3.877), (520, 3.613),
                      (550, 3.126), (600, 1.906), (650, 0.839), (700, 0.000)]

# Total energy against k-point spacing, referred to the finest mesh.
KPOINT_CONVERGENCE = [(0.50, 0.184), (0.40, 0.024), (0.35, 0.011),
                      (0.30, -0.002), (0.25, -0.003), (0.20, 0.000)]

# Spin states of bulk beta-CoOOH, eV relative to the low-spin ground state.
# The S = 2 initialisation relaxes into the S = 1 solution.
COOOH_SPIN_STATES = [("S = 0", 0.000), ("S = 1", 0.586), ("S = 2", 0.586)]

# beta-Co(OH)2 relaxed without symmetry constraints against the symmetrised
# cell, in eV per formula unit, with the corresponding band gap in eV.
JAHN_TELLER = [("P1", 0.00, 2.60), ("P-3m1", 0.82, 0.001)]

# The *O intermediate: energy relative to the unseeded solution, in eV, with
# the magnetic moment carried by the adsorbed oxygen in Bohr magnetons.
OXO_SOLUTIONS = [("unseeded", 0.000, -0.77), ("AFM oxo", -0.546, 0.00),
                 ("FM oxo", -1.138, 0.00)]

# Adsorption-site search, eV relative to the site used throughout.
SITE_SEARCH = [("*OH, Co2", 0.121), ("*OH, Co1", 0.265),
               ("H*, Co top", 1.659), ("*OH, Co next to Ni", -0.755)]

# ZIF-67 reflections from the relaxed framework model, degrees 2-theta.
XRD_CALCULATED = [7.39, 10.46, 12.83, 14.82, 16.58, 18.18]
