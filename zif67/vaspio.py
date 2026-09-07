"""Readers for VASP output.

All quantities used in the analysis are taken from the raw output files rather
than from a cached table, so that reported numbers cannot drift away from the
calculations that produced them.
"""
from __future__ import annotations
__author__ = "Nabil Khossossi"

import os

import numpy as np

EDIFFG = 0.02  # eV/A, the force criterion used throughout


def _path(root: str, directory: str, name: str) -> str:
    return os.path.join(root, directory, name)


def final_energy(root: str, directory: str) -> tuple[float, float]:
    """Return (E0, total magnetic moment) from the last ionic step."""
    last = None
    with open(_path(root, directory, "OSZICAR")) as fh:
        for line in fh:
            if " F= " in line:
                last = line.split()
    if last is None:
        raise ValueError(f"no ionic steps in {directory}/OSZICAR")
    energy = float(last[4])
    moment = float(last[-1]) if "mag=" in " ".join(last) else 0.0
    return energy, moment


def max_force(root: str, directory: str) -> float | None:
    """Largest force component on any unconstrained atom, in eV/A.

    Atoms held fixed by selective dynamics are excluded, and force blocks left
    truncated by an interrupted write are skipped.
    """
    text = open(_path(root, directory, "OUTCAR"), errors="ignore").read()
    contcar = _path(root, directory, "CONTCAR")
    if not os.path.exists(contcar):
        return None

    lines = open(contcar).read().splitlines()
    natoms = sum(int(x) for x in lines[6].split())
    selective = lines[7].strip().upper().startswith("S")
    flags = [l.split()[3:6] for l in lines[9:9 + natoms]] if selective else None
    free = [i for i in range(natoms)
            if flags is None or any(f.upper() == "T" for f in flags[i])]

    fmax = None
    out = text.splitlines()
    for i, line in enumerate(out):
        if "TOTAL-FORCE (eV/Angst)" in line:
            rows = out[i + 2:i + 2 + natoms]
            if len(rows) < natoms or any(len(r.split()) != 6 for r in rows):
                continue
            fmax = max(max(abs(float(v)) for v in rows[k].split()[3:6])
                       for k in free)
    return fmax


def converged(root: str, directory: str) -> bool:
    text = open(_path(root, directory, "OUTCAR"), errors="ignore").read()
    return "reached required accuracy" in text


def read_run(root: str, directory: str) -> dict:
    """Energy, moment, convergence flag and residual force for one directory."""
    energy, moment = final_energy(root, directory)
    return {
        "energy": energy,
        "moment": moment,
        "converged": converged(root, directory),
        "fmax": max_force(root, directory),
    }


def species(root: str, directory: str) -> np.ndarray:
    """Per-atom element labels, in POSCAR order."""
    lines = open(_path(root, directory, "CONTCAR")).read().splitlines()
    labels: list[str] = []
    for element, count in zip(lines[5].split(),
                              (int(x) for x in lines[6].split())):
        labels += [element] * count
    return np.array(labels)


def geometry(root: str, directory: str) -> tuple[np.ndarray, np.ndarray]:
    """Cartesian coordinates and element labels."""
    lines = open(_path(root, directory, "CONTCAR")).read().splitlines()
    scale = float(lines[1].split()[0])
    cell = np.array([[float(x) for x in lines[2 + i].split()]
                     for i in range(3)]) * scale
    counts = [int(x) for x in lines[6].split()]
    natoms = sum(counts)
    start = 9 if lines[7].strip().upper().startswith("S") else 8
    frac = np.array([[float(x) for x in lines[start + i].split()[:3]]
                     for i in range(natoms)])
    return frac @ cell, species(root, directory)


def _last_block(root: str, directory: str, header: str) -> np.ndarray | None:
    """Parse the final per-atom table introduced by `header` in OUTCAR."""
    lines = open(_path(root, directory, "OUTCAR"),
                 errors="ignore").read().splitlines()
    starts = [i for i, l in enumerate(lines) if l.strip().startswith(header)]
    if not starts:
        return None
    rows = []
    for line in lines[starts[-1] + 4:]:
        fields = line.split()
        if line.startswith("---"):
            continue
        if line.strip().startswith("tot") or not fields or not fields[0].isdigit():
            break
        rows.append([float(x) for x in fields[1:]])
    return np.array(rows)


def site_moments(root: str, directory: str) -> np.ndarray | None:
    """Per-atom magnetic moments from the LORBIT decomposition."""
    return _last_block(root, directory, "magnetization (x)")


def site_charges(root: str, directory: str) -> np.ndarray | None:
    """Per-atom sphere-projected (s, p, d, total) charges from LORBIT = 11.

    These are comparable within an element only: different elements use
    different PAW valence configurations, so their absolute sphere charges are
    not on a common scale.
    """
    return _last_block(root, directory, "total charge")


def read_doscar(root: str, directory: str):
    """Return (E - E_F, total DOS up, total DOS down, per-atom blocks, spin)."""
    lines = open(_path(root, directory, "DOSCAR")).read().splitlines()
    natoms = int(lines[0].split()[0])
    nedos = int(lines[5].split()[2])
    fermi = float(lines[5].split()[3])

    total = np.array([[float(x) for x in lines[6 + i].split()]
                      for i in range(nedos)])
    energy = total[:, 0] - fermi
    spin = total.shape[1] >= 5
    up = total[:, 1]
    down = total[:, 2] if spin else np.zeros(nedos)

    blocks, k = [], 6 + nedos
    for _ in range(natoms):
        if k >= len(lines):
            break
        k += 1  # each atom block is preceded by a repeat of the header line
        blocks.append(np.array([[float(x) for x in lines[k + i].split()]
                                for i in range(nedos)]))
        k += nedos
    return energy, up, down, blocks, spin
