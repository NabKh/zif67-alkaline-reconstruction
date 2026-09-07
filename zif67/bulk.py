"""Bulk thermodynamics: framework hydrolysis and Co/Ni mixing.

Two quantities are computed here.

Framework hydrolysis is written in charge- and mass-balanced neutral form,

    Co(mim)2 + 2 H2O -> beta-Co(OH)2 + 2 Hmim

Referring both the hydroxide and the methylimidazolate to their conjugate acids
gives the algebraically identical ionic form. Because the reaction consumes two
hydroxide ions and releases two methylimidazolate ions, the proton chemical
potential enters with equal and opposite coefficients and cancels, so the result
is independent of pH and of the electrode reference.

Mixing in the oxyhydroxide is referred to the two end-members per formula unit,

    dH_mix(x) = E[Co(1-x)Ni(x)OOH] - (1-x) E[CoOOH] - x E[NiOOH]

Each composition samples one ordered arrangement rather than a configurational
average, so the result establishes miscibility rather than a quantitative
mixing enthalpy.
"""
from __future__ import annotations
__author__ = "Nabil Khossossi"

from .vaspio import read_run

EV_TO_KJ_PER_MOL = 96.485

HYDROLYSIS = {
    "framework": "3_zif67/afm",          # antiferromagnetic ground state
    "hydroxide": "2_bulk/co_oh2/relax2",
    "linker": "1_molecules/hmim_solv",   # implicit solvation
    "water": "1_molecules/h2o",
}
FRAMEWORK_COBALT_COUNT = 6

MIXING = {
    0.00: "4_oer_slabs/mixing/x000",
    0.25: "4_oer_slabs/mixing/x025",
    0.50: "4_oer_slabs/mixing/x050",
    1.00: "4_oer_slabs/mixing/x100",
}
FORMULA_UNITS_PER_CELL = 4


def hydrolysis_energy(root: str) -> dict:
    """Electronic reaction energy per cobalt centre, in eV."""
    energies = {k: read_run(root, d)["energy"] for k, d in HYDROLYSIS.items()}
    delta = (energies["hydroxide"] * FRAMEWORK_COBALT_COUNT
             + 2 * FRAMEWORK_COBALT_COUNT * energies["linker"]
             - energies["framework"]
             - 2 * FRAMEWORK_COBALT_COUNT * energies["water"])
    return {"per_cobalt": delta / FRAMEWORK_COBALT_COUNT, "total": delta}


def mixing_enthalpies(root: str) -> list[dict]:
    """Mixing enthalpy per formula unit across the substitution series."""
    per_fu = {x: read_run(root, d)["energy"] / FORMULA_UNITS_PER_CELL
              for x, d in MIXING.items()}
    end_co, end_ni = per_fu[0.00], per_fu[1.00]

    results = []
    for x in sorted(per_fu):
        delta = per_fu[x] - (1 - x) * end_co - x * end_ni
        results.append({
            "x": x,
            "energy_per_fu": per_fu[x],
            "dH_mix_eV": delta,
            "dH_mix_kJ_per_mol": delta * EV_TO_KJ_PER_MOL,
        })
    return results


def report(root: str) -> None:
    hydrolysis = hydrolysis_energy(root)
    print("Framework hydrolysis, Co(mim)2 + 2 H2O -> beta-Co(OH)2 + 2 Hmim")
    print(f"  electronic reaction energy = "
          f"{hydrolysis['per_cobalt']:+.3f} eV per Co\n")

    print("Mixing in beta-Co(1-x)Ni(x)OOH")
    print(f"{'x':>6s} {'E per f.u. (eV)':>17s} {'dH_mix (kJ/mol)':>17s}")
    for row in mixing_enthalpies(root):
        print(f"{row['x']:6.2f} {row['energy_per_fu']:17.4f} "
              f"{row['dH_mix_kJ_per_mol']:17.1f}")
