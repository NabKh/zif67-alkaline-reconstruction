"""Oxygen evolution on beta-CoOOH(01-12).

Associative four-electron mechanism within the computational hydrogen
electrode (Norskov et al., 2004):

    dG1 = G(*OH)  - G(*)   - [G(H2O) - 1/2 G(H2)]
    dG2 = G(*O)   - G(*OH) + 1/2 G(H2)
    dG3 = G(*OOH) - G(*O)  - [G(H2O) - 1/2 G(H2)]
    dG4 = dG(O2)  - dG1 - dG2 - dG3
    eta = max(dG1..dG4)/e - 1.23 V

The *O intermediate is taken from an explicitly seeded Co(IV)=O solution. An
unseeded magnetic initialisation converges instead to an oxygen radical lying
1.14 eV higher, which would inflate the potential-determining step by more than
a volt; see the spin-state notes in the README.
"""
from __future__ import annotations
__author__ = "Nabil Khossossi"

import numpy as np

from . import thermo
from .vaspio import read_run

MOLECULES = {"H2": "1_molecules/h2", "H2O": "1_molecules/h2o"}

SURFACES = {
    "pristine": {
        "*":    "4_oer_slabs/clean_ls",
        "*OH":  "4_oer_slabs/oh_ls",
        "*O":   "4_oer_slabs/u_sensitivity/oxo_fm_o",
        "*OOH": "4_oer_slabs/ooh_ls",
    },
}

STATES = ("*", "*OH", "*O", "*OOH")


def free_energies(root: str, surface: str = "pristine") -> dict:
    """Step free energies, overpotential and the underlying state energies."""
    molecules = {name: thermo.gibbs(read_run(root, directory)["energy"], name)
                 for name, directory in MOLECULES.items()}

    reference = thermo.proton_electron_reference(molecules["H2O"],
                                                 molecules["H2"])
    half_h2 = 0.5 * molecules["H2"]

    states = {}
    for state in STATES:
        run = read_run(root, SURFACES[surface][state])
        states[state] = {
            "energy": run["energy"],
            "gibbs": thermo.gibbs(run["energy"], state),
            "moment": run["moment"],
            "converged": run["converged"],
            "fmax": run["fmax"],
        }

    g = {s: states[s]["gibbs"] for s in STATES}
    steps = [
        g["*OH"] - g["*"] - reference,
        g["*O"] - g["*OH"] + half_h2,
        g["*OOH"] - g["*O"] - reference,
    ]
    steps.append(thermo.G_O2 - sum(steps))

    limiting = int(np.argmax(steps))
    return {
        "states": states,
        "steps": steps,
        "potential_determining_step": limiting + 1,
        "overpotential": max(steps) - thermo.U_EQUILIBRIUM,
        "converged": all(states[s]["converged"] for s in STATES),
    }


def scaling_descriptors(steps: list[float]) -> dict:
    """Quantities used for the scaling relation and the activity volcano.

    dG(OOH) - dG(OH) is compared with the universal value of 3.2 +/- 0.2 eV
    (Man et al. 2011); dG(O) - dG(OH) is the volcano descriptor.
    """
    dg_oh = steps[0]
    dg_o = steps[0] + steps[1]
    dg_ooh = dg_o + steps[2]
    return {
        "dG_OH": dg_oh,
        "dG_O": dg_o,
        "dG_OOH": dg_ooh,
        "scaling": dg_ooh - dg_oh,
        "descriptor": dg_o - dg_oh,
    }


def report(root: str) -> None:
    result = free_energies(root)
    labels = ["* -> *OH", "*OH -> *O", "*O -> *OOH", "*OOH -> O2"]

    print("Oxygen evolution on beta-CoOOH(01-12)\n")
    print(f"{'state':6s} {'E0 (eV)':>13s} {'G (eV)':>13s} {'moment':>8s} "
          f"{'Fmax':>7s}")
    for state in STATES:
        s = result["states"][state]
        flag = "" if s["converged"] else "  not converged"
        print(f"{state:6s} {s['energy']:13.5f} {s['gibbs']:13.5f} "
              f"{s['moment']:8.2f} {s['fmax']:7.4f}{flag}")

    print()
    for i, (label, value) in enumerate(zip(labels, result["steps"]), start=1):
        mark = "  <- potential-determining" \
            if i == result["potential_determining_step"] else ""
        print(f"dG{i}  {label:12s} = {value:6.3f} eV{mark}")

    d = scaling_descriptors(result["steps"])
    print(f"\ndG(OOH) - dG(OH) = {d['scaling']:.3f} eV "
          f"(universal relation 3.2 +/- 0.2)")
    print(f"overpotential    = {result['overpotential']:.3f} V")
