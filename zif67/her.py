"""Hydrogen evolution on the reconstructed cobalt phases.

Hydrogen adsorption free energies follow

    dG(H*) = E(slab + H) - E(slab) - 1/2 E(H2) + 0.24 eV

where the composite correction collects the zero-point and entropy terms
(Norskov et al., J. Electrochem. Soc. 152, J23 (2005)). At half a monolayer the
differential value is reported, in which the reference slab already carries one
hydrogen atom.

The alkaline Volmer step proceeds by dissociation of water rather than by
discharge of a hydronium ion, so its reaction energy is reported alongside.
"""
from __future__ import annotations
__author__ = "Nabil Khossossi"

from . import thermo
from .vaspio import read_run

HYDROGEN = "1_molecules/h2"

SURFACES = [
    # label, clean slab, hydrogenated slab, coverage, differential?
    ("beta-CoOOH(01-12)", "4_oer_slabs/clean_ls",
     "5_her_slabs/cooh_0112_h", "0.25 ML", False),
    ("beta-Co(OH)2(001)", "5_her_slabs/co_oh2_001_clean",
     "5_her_slabs/co_oh2_001_h_025ml", "0.25 ML", False),
    ("beta-Co(OH)2(001)", "5_her_slabs/co_oh2_001_h_025ml",
     "5_her_slabs/co_oh2_001_h_050ml", "0.50 ML", True),
]

WATER_DISSOCIATION = ("5_her_slabs/neb_water_dissociation/relax_00",
                      "5_her_slabs/neb_water_dissociation/relax_06")


def adsorption_energies(root: str) -> list[dict]:
    half_h2 = 0.5 * read_run(root, HYDROGEN)["energy"]
    results = []
    for label, clean, loaded, coverage, differential in SURFACES:
        bare = read_run(root, clean)
        with_h = read_run(root, loaded)
        results.append({
            "surface": label,
            "coverage": coverage,
            "differential": differential,
            "dG_H": (with_h["energy"] - bare["energy"] - half_h2
                     + thermo.H_ADSORPTION_CORRECTION),
            "converged": bare["converged"] and with_h["converged"],
        })
    return results


def water_dissociation(root: str) -> dict:
    """Reaction energy for H2O* -> OH* + H* on the reconstructed surface."""
    initial, final = (read_run(root, d) for d in WATER_DISSOCIATION)
    return {
        "reaction_energy": final["energy"] - initial["energy"],
        "converged": initial["converged"] and final["converged"],
    }


def report(root: str) -> None:
    print("Hydrogen adsorption\n")
    print(f"{'surface':22s} {'coverage':10s} {'dG(H*) (eV)':>12s}")
    for r in adsorption_energies(root):
        note = " differential" if r["differential"] else ""
        flag = "" if r["converged"] else "  not converged"
        print(f"{r['surface']:22s} {r['coverage']:10s} "
              f"{r['dG_H']:12.3f}{note}{flag}")

    water = water_dissociation(root)
    print(f"\nH2O* -> OH* + H*  =  {water['reaction_energy']:+.3f} eV")
