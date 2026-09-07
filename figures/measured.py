"""Measured and literature values used in the figures.

Everything here comes from experiment or from the literature, not from the
calculations in this repository. Computed quantities are obtained from the
`zif67` package at plotting time.
"""
__author__ = "Nabil Khossossi"

# Lattice parameters from diffraction, in Angstrom.
#   Co3O4:        ICSD, normal spinel
#   beta-Co(OH)2: brucite-type, P-3m1
#   beta-CoOOH:   R-3m, hexagonal setting
#   Co (hcp):     room-temperature value
LATTICE_EXPERIMENT = {
    ("Co3O4", "a"): 8.084,
    ("beta-Co(OH)2", "a"): 3.183,
    ("beta-Co(OH)2", "c"): 4.653,
    ("beta-CoOOH", "a"): 2.855,
    ("beta-CoOOH", "c"): 13.150,
    ("Co", "a"): 2.507,
}

# ZIF-67 reflections observed here, degrees 2-theta, Cu K-alpha.
XRD_OBSERVED = [7.30, 10.40, 12.70, 14.80, 16.40, 18.00]

# SEM-EDX atomic ratios of the ZIF-67 D@NF electrode.
# The pristine Ni signal originates from the exposed foam substrate and is
# used as the blank against which the post-operation values are read.
EDX_N_OVER_CO = {r"ideal Co(mim)$_2$": 4.000, "pristine": 3.677,
                 "after HER": 0.221, "after OER": 0.127}
EDX_NI_OVER_CO = {"pristine": 1.045, "after HER": 0.513, "after OER": 1.431}

# Nickel fraction of the post-OER layer, bracketed between the blank-corrected
# lower bound and the raw signal.
EDX_COMPOSITION_RANGE = (0.25, 0.59)

# Charge-transfer resistance from impedance spectroscopy, ohm.
CHARGE_TRANSFER_RESISTANCE = {"pristine": 1084, "after HER": 92,
                              "after OER": 122}

# Universal scaling relation between the OOH and OH adsorption free energies,
# Man et al., ChemCatChem 3, 1159 (2011).
SCALING_RELATION = 3.2
SCALING_SPREAD = 0.2
