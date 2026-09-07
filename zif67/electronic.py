"""Electronic-structure descriptors.

Band gaps and projected densities of states are read from DOSCAR; the oxidation
state of the active metal is followed through its LORBIT sphere charges, its
local magnetic moment and its metal-oxygen bond length.

Band centres are first moments of the projected density of states over the
valence window, referred to the Fermi level. The window must be restricted to
the band of interest: integrating over the full DOSCAR range admits high-lying
free-electron states carrying a small spurious projection and displaces the
centre by several electronvolts.

Gaps below roughly 0.5 eV lie at the resolution of the energy grid on which the
density of states was accumulated and should be treated as upper bounds.
"""
from __future__ import annotations
__author__ = "Nabil Khossossi"

import numpy as np

from .vaspio import (read_doscar, geometry, species, site_charges,
                     site_moments)

VALENCE_WINDOW = (-10.0, 6.0)
OXYGEN_WINDOW = (-10.0, 4.0)

PHASES = [
    ("ZIF-67",            "3_zif67/afm"),
    ("beta-Co(OH)2",      "2_bulk/co_oh2/relax2"),
    ("beta-CoOOH",        "4_oer_slabs/mixing/x000"),
    ("Co0.75Ni0.25OOH",   "4_oer_slabs/mixing/x025"),
    ("Co0.5Ni0.5OOH",     "4_oer_slabs/mixing/x050"),
    ("beta-NiOOH",        "4_oer_slabs/mixing/x100"),
]

CYCLE = [
    ("*",    "4_oer_slabs/clean_ls"),
    ("*OH",  "4_oer_slabs/oh_ls"),
    ("*O",   "4_oer_slabs/u_sensitivity/oxo_fm_o"),
    ("*OOH", "4_oer_slabs/ooh_ls"),
]


def _orbital_columns(block: np.ndarray, orbital: str, spin: bool) -> list[int]:
    """Column indices of the requested orbital, spin channels interleaved."""
    ncols = block.shape[1] - 1
    if orbital == "d":
        columns = range(9, 19) if (spin and ncols >= 18) else range(5, 7)
    else:  # p
        columns = range(3, 9) if (spin and ncols >= 18) else range(2, 5)
    return [c for c in columns if c < block.shape[1]]


def band_gap(energy: np.ndarray, up: np.ndarray, down: np.ndarray,
             tol: float = 1e-3) -> float:
    """Gap straddling the Fermi level; zero for a metal."""
    total = up + np.abs(down)
    occupied = energy <= 0
    if not occupied.any() or occupied.all():
        return 0.0
    valence = energy[occupied][np.where(total[occupied] > tol)[0]]
    conduction = energy[~occupied][np.where(total[~occupied] > tol)[0]]
    if len(valence) == 0 or len(conduction) == 0:
        return 0.0
    return max(0.0, conduction.min() - valence.max())


def band_centre(energy: np.ndarray, density: np.ndarray,
                window: tuple[float, float]) -> float | None:
    inside = (energy >= window[0]) & (energy <= window[1])
    if density[inside].sum() <= 0:
        return None
    return float(np.trapezoid(energy[inside] * density[inside], energy[inside])
                 / np.trapezoid(density[inside], energy[inside]))


def projected_dos(root: str, directory: str, element: str, orbital: str):
    """Spin-resolved projected DOS summed over all atoms of one element."""
    energy, _, _, blocks, spin = read_doscar(root, directory)
    labels = species(root, directory)
    indices = [i for i, l in enumerate(labels)
               if l == element and i < len(blocks)]
    if not indices:
        return None
    up = np.zeros_like(energy)
    down = np.zeros_like(energy)
    for i in indices:
        columns = _orbital_columns(blocks[i], orbital, spin)
        up += blocks[i][:, columns[0::2]].sum(axis=1)
        if spin:
            down += blocks[i][:, columns[1::2]].sum(axis=1)
    return energy, up, down


def phase_descriptors(root: str) -> list[dict]:
    """Band gap and band centres for each phase in the reconstruction series."""
    results = []
    for label, directory in PHASES:
        energy, up, down, blocks, spin = read_doscar(root, directory)
        entry = {"phase": label, "gap": band_gap(energy, up, down)}
        for element in ("Co", "Ni"):
            projection = projected_dos(root, directory, element, "d")
            entry[f"d_centre_{element}"] = (
                band_centre(energy, projection[1] + projection[2],
                            VALENCE_WINDOW) if projection else None)
        oxygen = projected_dos(root, directory, "O", "p")
        entry["O2p_centre"] = (band_centre(energy, oxygen[1] + oxygen[2],
                                           OXYGEN_WINDOW) if oxygen else None)
        results.append(entry)
    return results


def active_site(root: str, directory: str) -> tuple[int, str, float]:
    """Metal bonded to the adsorbate, and that bond length.

    For *OOH the metal-bound oxygen is the lower of the two, so the highest
    oxygen alone would identify the wrong end. Every oxygen within 2 A of the
    topmost one is considered and the shortest metal-oxygen contact taken.
    """
    positions, labels = geometry(root, directory)
    metals = [i for i, l in enumerate(labels) if l in ("Co", "Ni")]
    oxygens = [i for i, l in enumerate(labels) if l == "O"]
    top = max(positions[i, 2] for i in oxygens)
    candidates = [i for i in oxygens if positions[i, 2] > top - 2.0]
    distance, metal, _ = min(
        (np.linalg.norm(positions[m] - positions[o]), m, o)
        for o in candidates for m in metals)
    return metal, labels[metal], distance


def redox_cycle(root: str) -> list[dict]:
    """Oxidation state of the active metal through the four intermediates."""
    results = []
    for label, directory in CYCLE:
        index, element, distance = active_site(root, directory)
        charges = site_charges(root, directory)
        moments = site_moments(root, directory)
        results.append({
            "state": label,
            "element": element,
            "index": index,
            "bond_length": distance,
            "d_occupancy": charges[index, 2] if charges is not None else None,
            "moment": moments[index, -1] if moments is not None else None,
        })
    return results


def site_dos(root: str, directory: str, index: int, orbital: str = "d"):
    """Spin-resolved projected DOS on one atom."""
    energy, _, _, blocks, spin = read_doscar(root, directory)
    columns = _orbital_columns(blocks[index], orbital, spin)
    return (energy,
            blocks[index][:, columns[0::2]].sum(axis=1),
            blocks[index][:, columns[1::2]].sum(axis=1))


def report(root: str) -> None:
    print("Electronic structure of the reconstruction series\n")
    print(f"{'phase':20s} {'gap (eV)':>9s} {'e_d Co':>8s} {'e_d Ni':>8s} "
          f"{'e_O2p':>8s}")
    for row in phase_descriptors(root):
        fmt = lambda v: f"{v:8.2f}" if v is not None else f"{'-':>8s}"
        print(f"{row['phase']:20s} {row['gap']:9.2f} "
              f"{fmt(row['d_centre_Co'])} {fmt(row['d_centre_Ni'])} "
              f"{fmt(row['O2p_centre'])}")

    print("\nRedox state of the active metal through the OER cycle\n")
    print(f"{'state':6s} {'metal':>6s} {'M-O (A)':>9s} {'d occ.':>8s} "
          f"{'moment':>8s}")
    for row in redox_cycle(root):
        print(f"{row['state']:6s} {row['element']:>6s} "
              f"{row['bond_length']:9.3f} {row['d_occupancy']:8.3f} "
              f"{row['moment']:8.3f}")
