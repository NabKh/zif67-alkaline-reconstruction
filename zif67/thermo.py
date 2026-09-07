"""Free-energy corrections and electrochemical references.

Gibbs energies are formed as G = E_DFT + ZPE - TS at 298.15 K. Molecular
corrections come from finite-difference Hessians computed here;
adsorbate corrections are the standard tabulation of Man et al., ChemCatChem 3,
1159 (2011), which is transferable between related oxide and oxyhydroxide
surfaces and is common to every surface compared here.
"""
from __future__ import annotations
__author__ = "Nabil Khossossi"

TEMPERATURE = 298.15  # K

# (ZPE, T*S) in eV
CORRECTIONS = {
    "H2":   (0.268, 0.403),
    "H2O":  (0.568, 0.670),
    "*":    (0.000, 0.000),
    "*OH":  (0.350, 0.010),
    "*O":   (0.050, 0.000),
    "*OOH": (0.400, 0.010),
}

# Composite ZPE and entropy correction for adsorbed hydrogen referenced to
# 1/2 H2(g); Norskov et al., J. Electrochem. Soc. 152, J23 (2005).
H_ADSORPTION_CORRECTION = 0.24

# Experimental free energy of 2 H2O -> O2 + 2 H2, imposed rather than computed
# because semi-local functionals overbind molecular oxygen.
G_O2 = 4.92

# Equilibrium potential of the four-electron oxygen evolution reaction.
U_EQUILIBRIUM = 1.23


def gibbs(energy: float, species: str) -> float:
    """G = E + ZPE - TS for a species with tabulated corrections."""
    zpe, ts = CORRECTIONS[species]
    return energy + zpe - ts


def proton_electron_reference(g_water: float, g_hydrogen: float) -> float:
    """Chemical potential of a removed proton-electron pair.

    Within the computational hydrogen electrode the pair is referred to gaseous
    H2 at the reversible hydrogen electrode. In alkaline media the hydroxide
    potential follows from the water equilibrium, so mu(H+) cancels between the
    two sides of every step and the result is independent of pH.
    """
    return g_water - 0.5 * g_hydrogen
