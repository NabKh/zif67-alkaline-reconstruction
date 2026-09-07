#!/usr/bin/env python3
"""Figures.

Each panel is a function of a single axes object, so the multi-panel composites
and the standalone single-panel versions are generated from identical code and
cannot diverge.

Computed quantities are read from the `zif67` package; measured and literature
values are collected in `measured.py`.

    python figures/make_figures.py --root /path/to/calculations
"""
from __future__ import annotations
__author__ = "Nabil Khossossi"

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from style import (apply_style, framed, panel_label, no_minor, save,
                   C_PRIMARY, C_SECONDARY, C_TERTIARY, C_QUAT,
                   INK, SUBTLE, COL_SINGLE, COL_DOUBLE)
from zif67 import bulk, electronic, her, oer

import benchmarks
import measured

apply_style(7.0)

ROOT = None          # calculation tree, set from the command line
OUT_C = "figures/composite"
OUT_P = "figures/panels"

_CACHE = {}


def data():
    """Compute every quantity the figures need, once."""
    if not _CACHE:
        result = oer.free_energies(ROOT)
        _CACHE["oer"] = result
        _CACHE["descriptors"] = oer.scaling_descriptors(result["steps"])
        _CACHE["her"] = her.adsorption_energies(ROOT)
        _CACHE["water"] = her.water_dissociation(ROOT)
        _CACHE["hydrolysis"] = bulk.hydrolysis_energy(ROOT)
        _CACHE["mixing"] = bulk.mixing_enthalpies(ROOT)
        _CACHE["phases"] = electronic.phase_descriptors(ROOT)
        _CACHE["cycle"] = electronic.redox_cycle(ROOT)
    return _CACHE





def _sequence():
    """Phase label, band gap, Co and Ni d-band centres, O 2p band centre."""
    label = {"ZIF-67": "ZIF-67",
             "beta-Co(OH)2": r"$\beta$-Co(OH)$_2$",
             "beta-CoOOH": r"$\beta$-CoOOH",
             "Co0.75Ni0.25OOH": "Co$_{0.75}$Ni$_{0.25}$OOH",
             "Co0.5Ni0.5OOH": "Co$_{0.5}$Ni$_{0.5}$OOH",
             "beta-NiOOH": r"$\beta$-NiOOH"}
    return [(label[r["phase"]], r["gap"], r["d_centre_Co"], r["d_centre_Ni"],
             r["O2p_centre"]) for r in data()["phases"]]


def _pdos(directory):
    """Element- and orbital-projected DOS for one structure."""
    projections = {}
    energy = None
    for element, orbital in (("Co", "d"), ("Ni", "d"), ("O", "p")):
        result = electronic.projected_dos(ROOT, directory, element, orbital)
        if result is None:
            continue
        energy, up, down = result
        projections[element] = (up, down)
    e, u, dn, _, _ = electronic.read_doscar(ROOT, directory)
    return energy, projections, electronic.band_gap(e, u, dn)



def _phase_dos_series(labels):
    """Element-projected DOS and band gap for selected phases, in order."""
    directory = {r"$\beta$-Co(OH)$_2$": "2_bulk/co_oh2/relax2",
                 r"$\beta$-CoOOH": "4_oer_slabs/mixing/x000",
                 "Co$_{0.5}$Ni$_{0.5}$OOH": "4_oer_slabs/mixing/x050",
                 r"$\beta$-NiOOH": "4_oer_slabs/mixing/x100"}
    series = []
    for label in labels:
        energy, projections, gap = _pdos(directory[label])
        series.append((label, energy, projections, gap))
    return series


def _cycle_dos():
    """d-projected DOS on the active metal for each OER intermediate."""
    series = []
    for row in data()["cycle"]:
        directory = dict(electronic.CYCLE)[row["state"]]
        energy, up, down = electronic.site_dos(ROOT, directory, row["index"])
        series.append((row["state"], row["element"], row["index"],
                       energy, up, down))
    return series


def load():
    """Populate the panel data from the package, the measurements and the
    benchmark tests. Called once, after the calculation root is known."""
    global HYDROLYSIS, LATTICE, MIX, NCO, OER_DG, OER_ETA, DGH, WD
    global ENCUT, KSPACING, SPIN_COOOH, OXO, SITES

    d = data()

    dh = d["hydrolysis"]["per_cobalt"]
    HYDROLYSIS = [(r"$\Delta E$", dh),
                  ("$\\Delta G$\nharm.", -0.071),
                  ("$\\Delta G$\nqRRHO", -0.100)]

    LATTICE = [("Co$_3$O$_4$ $a$", 8.089, measured.LATTICE_EXPERIMENT[("Co3O4", "a")], True),
               (r"$\beta$-Co(OH)$_2$ $a$", 3.181, measured.LATTICE_EXPERIMENT[("beta-Co(OH)2", "a")], True),
               (r"$\beta$-CoOOH $a$", 2.858, measured.LATTICE_EXPERIMENT[("beta-CoOOH", "a")], True),
               ("Co (hcp) $a$", 2.473, measured.LATTICE_EXPERIMENT[("Co", "a")], True),
               (r"$\beta$-Co(OH)$_2$ $c$", 4.545, measured.LATTICE_EXPERIMENT[("beta-Co(OH)2", "c")], False),
               (r"$\beta$-CoOOH $c$", 12.990, measured.LATTICE_EXPERIMENT[("beta-CoOOH", "c")], False)]

    MIX = [(row["x"], row["dH_mix_kJ_per_mol"]) for row in d["mixing"]]

    NCO = [(k, v) for k, v in measured.EDX_N_OVER_CO.items()]

    OER_DG = d["oer"]["steps"]
    OER_ETA = d["oer"]["overpotential"]

    DGH = [(r"$\beta$-CoOOH" + "\n0.25 ML", d["her"][0]["dG_H"]),
           (r"$\beta$-Co(OH)$_2$" + "\n0.25 ML", d["her"][1]["dG_H"]),
           (r"$\beta$-Co(OH)$_2$" + "\n0.50 ML", d["her"][2]["dG_H"])]

    WD = [("H$_2$O*", 0.000), ("OH* + H*", d["water"]["reaction_energy"])]

    ENCUT = benchmarks.CUTOFF_CONVERGENCE
    KSPACING = benchmarks.KPOINT_CONVERGENCE
    SPIN_COOOH = benchmarks.COOOH_SPIN_STATES
    OXO = [(l, e, f"{m:+.2f}".replace("+0.00", "0.00"))
           for l, e, m in benchmarks.OXO_SOLUTIONS]
    SITES = benchmarks.SITE_SEARCH


# ================================================================== panels ===
def panel_hydrolysis(ax):
    """Framework hydrolysis as a reaction level diagram.

    Levels rather than bars: the quantity is a reaction energy between two
    defined states, and three estimates of the same product level are shown.
    """
    labels = [r"$\Delta E$", r"$\Delta G$, harmonic", r"$\Delta G$, quasi-RRHO"]
    values = [v for _, v in HYDROLYSIS]
    styles = [(C_PRIMARY, "-"), (C_PRIMARY, (0, (4, 2))), (C_SECONDARY, "-")]

    ax.plot([-0.30, 0.30], [0, 0], color=INK, lw=2.0, solid_capstyle="round",
            zorder=3)
    for (value, label, (colour, dash)) in zip(values, labels, styles):
        ax.plot([0.70, 1.30], [value, value], color=colour, lw=2.0, ls=dash,
                solid_capstyle="round", zorder=3)
        ax.plot([0.30, 0.70], [0, value], color=colour, lw=0.6,
                ls=(0, (2, 2)), zorder=2)
        ax.text(1.36, value, f"{value:+.3f}  {label}", fontsize=6.5,
                va="center", ha="left")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Co(mim)$_2$\n+ 2 H$_2$O",
                        r"$\beta$-Co(OH)$_2$" + "\n+ 2 Hmim"])
    ax.set_ylabel("Reaction energy (eV per Co)")
    ax.set_xlim(-0.45, 2.55)
    ax.set_ylim(-0.135, 0.030)
    ax.yaxis.set_major_locator(MultipleLocator(0.04))
    no_minor(ax, "x"); framed(ax)


def panel_lattice_deviation(ax):
    d = [(l, 100 * (c - e) / e, ip) for l, c, e, ip in LATTICE]
    ys = np.arange(len(d))[::-1]
    ax.axvline(0, color=INK, lw=0.7, zorder=1)
    for y, (l, dv, ip) in zip(ys, d):
        ax.barh(y, dv, height=0.5, zorder=2, edgecolor="none",
                color=C_PRIMARY if ip else C_SECONDARY)
        ax.text(dv + (0.09 if dv >= 0 else -0.09), y, f"{dv:+.2f}",
                va="center", ha="left" if dv >= 0 else "right", fontsize=7)
    ax.set_yticks(ys); ax.set_yticklabels([x[0] for x in d])
    ax.set_xlabel("Deviation from experiment (%)")
    ax.set_xlim(-3.5, 1.6); ax.set_ylim(-0.95, len(d) - 0.25)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.legend(handles=[Patch(facecolor=C_PRIMARY, label="in-plane"),
                       Patch(facecolor=C_SECONDARY, label="interlayer")],
              loc="lower left", bbox_to_anchor=(0.0, -0.02))
    no_minor(ax, "y"); framed(ax); ax.tick_params(axis="y", length=0)


def panel_mixing_enthalpy(ax):
    """Formation energy of the mixed oxyhydroxide against composition.

    Points are plotted as markers only: each composition is one ordered
    arrangement, not a configurational average, so a connecting curve would
    imply a continuous function that the data do not support.
    """
    ax.axvspan(0.25, 0.59, color=C_TERTIARY, alpha=0.12, lw=0, zorder=0)
    ax.axhline(0, color=INK, lw=0.8, zorder=2)
    ax.text(-0.05, 0.55, "ideal mixing", fontsize=6.5, color=INK, va="bottom",
            ha="left")
    ax.text(0.42, 5.9, "EDX composition", ha="center", va="center",
            fontsize=6.5, color=SUBTLE)
    for x, y in MIX:
        end = x in (0.0, 1.0)
        ax.plot(x, y, "o", ms=6.5, mfc="white" if end else C_PRIMARY,
                mec=C_PRIMARY, mew=1.1, zorder=4)
    ax.text(0.0, -1.5, "CoOOH", fontsize=6.5, ha="left", va="top", color=SUBTLE)
    ax.text(1.0, -1.5, "NiOOH", fontsize=6.5, ha="right", va="top", color=SUBTLE)
    ax.annotate(r"$+$0.3", xy=(0.25, 0.3), xytext=(0.285, 2.9), fontsize=7,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=SUBTLE))
    ax.annotate(r"$-$11.1 kJ mol$^{-1}$", xy=(0.50, -11.1), xytext=(0.615, -8.2),
                fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.6, color=SUBTLE))
    ax.text(-0.055, -7.5, "mixing favorable", fontsize=6.3, color=SUBTLE,
            rotation=90, va="center")
    ax.set_xlabel(r"$x$ in $\beta$-Co$_{1-x}$Ni$_x$OOH")
    ax.set_ylabel(r"$\Delta H_{\mathrm{mix}}$ (kJ mol$^{-1}$ per f.u.)")
    ax.set_xlim(-0.09, 1.09); ax.set_ylim(-14.5, 7.2)
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    framed(ax)


def panel_linker_loss(ax):
    xs = np.arange(len(NCO))
    v = [x[1] for x in NCO]
    ax.bar(xs, v, width=0.46, edgecolor="none",
           color=[SUBTLE, C_PRIMARY, C_SECONDARY, C_TERTIARY])
    for x, y in zip(xs, v):
        ax.text(x, y + 0.10, f"{y:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs); ax.set_xticklabels([x[0] for x in NCO])
    ax.set_ylabel("N / Co atomic ratio")
    ax.set_ylim(0, 4.85)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.annotate("", xy=(3, 0.55), xytext=(1.15, 3.45),
                arrowprops=dict(arrowstyle="->", lw=0.8, color=SUBTLE,
                                connectionstyle="arc3,rad=-0.22"))
    ax.text(2.15, 2.45, u"94–96%\nlinker lost", ha="center", va="center",
            fontsize=7, color=SUBTLE)
    no_minor(ax, "x"); framed(ax)


def panel_oer_diagram(ax):
    lab = ["*", "*OH", "*O", "*OOH", r"* + O$_2$"]
    G0 = np.concatenate([[0], np.cumsum(OER_DG)])
    G1 = G0 - 1.23 * np.arange(5)
    for G, col, nm in ((G0, C_PRIMARY, "$U$ = 0 V"),
                       (G1, C_SECONDARY, "$U$ = 1.23 V")):
        for i, g in enumerate(G):
            ax.plot([i - 0.29, i + 0.29], [g, g], color=col, lw=1.8,
                    solid_capstyle="round", zorder=3)
        for i in range(4):
            ax.plot([i + 0.29, i + 0.71], [G[i], G[i + 1]], color=col, lw=0.6,
                    ls=(0, (2.5, 2)), zorder=2)
        ax.plot([], [], color=col, lw=1.8, label=nm)
    # potential-determining step: guide lines out to a clear column, then a
    # full-height double arrow with the label beside it (never on top of it)
    k = int(np.argmax(OER_DG))
    xa = 4.05
    for lev in (G1[k], G1[k + 1]):
        ax.plot([k + 0.29, xa], [lev, lev], color=INK, lw=0.5,
                ls=(0, (1.5, 1.5)), zorder=1)
    ax.annotate("", xy=(xa, G1[k + 1]), xytext=(xa, G1[k]),
                arrowprops=dict(arrowstyle="<|-|>", lw=0.9, color=INK,
                                mutation_scale=7, shrinkA=0, shrinkB=0))
    ax.text(xa + 0.18, 0.5 * (G1[k] + G1[k + 1]),
            rf"$\eta$ = {OER_ETA:.2f} V", fontsize=7.5, va="center", ha="left")
    ax.set_xticks(range(5)); ax.set_xticklabels(lab)
    ax.set_ylabel("Free energy (eV)")
    ax.set_xlim(-0.6, 5.9); ax.set_ylim(-1.45, 5.75)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.legend(loc="upper left")
    no_minor(ax, "x"); framed(ax)


def panel_hydrogen_binding(ax):
    xs = np.arange(len(DGH))
    v = [x[1] for x in DGH]
    ax.axhline(0, color=C_TERTIARY, lw=1.0, zorder=1)
    ax.text(len(DGH) - 0.55, -0.30, r"optimum", ha="right", va="center",
            fontsize=7, color=SUBTLE)
    ax.bar(xs, v, width=0.46, edgecolor="none", zorder=2,
           color=[C_SECONDARY, C_PRIMARY, C_PRIMARY])
    for x, y in zip(xs, v):
        ax.text(x, y + 0.08, f"+{y:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs); ax.set_xticklabels([x[0] for x in DGH], fontsize=7)
    ax.set_ylabel(r"$\Delta G_{\mathrm{H}^*}$ (eV)")
    ax.set_ylim(-0.55, 2.85)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    no_minor(ax, "x"); framed(ax)


def panel_water_dissociation(ax):
    for i, (l, g) in enumerate(WD):
        ax.plot([i - 0.24, i + 0.24], [g, g], color=C_PRIMARY, lw=1.8,
                solid_capstyle="round", zorder=3)
    ax.plot([0.24, 0.76], [WD[0][1], WD[1][1]], color=C_PRIMARY, lw=0.6,
            ls=(0, (2.5, 2)), zorder=2)
    ax.annotate("", xy=(1.0, WD[1][1]), xytext=(1.0, WD[0][1]),
                arrowprops=dict(arrowstyle="<->", lw=0.8, color=INK))
    ax.text(1.08, 0.5 * WD[1][1], "+0.30 eV", fontsize=7.5, va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels([x[0] for x in WD])
    ax.set_ylabel("Reaction energy (eV)")
    ax.set_xlim(-0.55, 1.75); ax.set_ylim(-0.10, 0.45)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    no_minor(ax, "x"); framed(ax)


def panel_pdos_pair(ax):
    names = [r"$\\beta$-CoOOH", "Co$_{0.5}$Ni$_{0.5}$OOH"]
    directories = {names[0]: "4_oer_slabs/mixing/x000",
                   names[1]: "4_oer_slabs/mixing/x050"}
    for nm, base in ((names[0], 0.0), (names[1], -1.12)):
        E, proj, gap = _pdos(directories[nm])
        w = (E > -8) & (E < 5)
        for el, col in (("Co", C_PRIMARY), ("Ni", C_QUAT), ("O", C_SECONDARY)):
            if el not in proj:
                continue
            u, dn = proj[el]
            t = (u + dn)[w]
            t = t / max(t.max(), 1e-9) * 0.82
            ax.fill_between(E[w], base, base + t, color=col, alpha=0.45, lw=0)
            ax.plot(E[w], base + t, color=col, lw=0.7)
    ax.axvline(0, color=INK, lw=0.7, ls=(0, (3, 2)))
    ax.text(0.16, 1.02, r"$E_\mathrm{F}$", fontsize=7.5, va="top")
    ax.text(-7.7, 1.02, r"$\beta$-CoOOH", fontsize=7.5, va="top")
    ax.text(-7.7, -0.10, r"Co$_{0.5}$Ni$_{0.5}$OOH", fontsize=7.5, va="top")
    for lab, col, y in (("Co 3$d$", C_PRIMARY, 0.66), ("O 2$p$", C_SECONDARY, 0.47),
                        ("Ni 3$d$", C_QUAT, 0.28)):
        ax.plot([3.0, 3.6], [y, y], color=col, lw=2.2, solid_capstyle="butt")
        ax.text(3.75, y, lab, fontsize=7, va="center")
    ax.set_xlim(-8, 5.4); ax.set_ylim(-1.30, 1.10)
    ax.set_yticks([])
    ax.set_xlabel(r"$E - E_\mathrm{F}$ (eV)")
    ax.set_ylabel("Projected DOS (arb. units)")
    no_minor(ax, "y"); framed(ax); ax.tick_params(axis="y", length=0)



def panel_redox_geometry(ax):
    cyc = data()["cycle"]
    xs = np.arange(len(cyc))
    dist = [c["bond_length"] for c in cyc]
    mom = [abs(c["moment"]) for c in cyc]
    ax.plot(xs, dist, "-o", color=C_PRIMARY, mfc=C_PRIMARY, mec="white",
            mew=0.7, ms=4.6)
    for x, v in zip(xs, dist):
        ax.text(x, v + 0.030, f"{v:.2f}", ha="center", va="bottom", fontsize=7,
                color=INK)
    for x, m in zip(xs, mom):
        ax.text(x, 1.552, f"{m:.2f}", ha="center", va="center", fontsize=7,
                color=SUBTLE)
    ax.text(-0.36, 1.596, r"local moment ($\mu_\mathrm{B}$)", fontsize=6.5,
            color=SUBTLE, ha="left")
    ax.text(-0.30, 1.955, "Co(III)", fontsize=7)
    ax.text(1.75, 1.955, "Co(IV)", fontsize=7)
    ax.annotate("", xy=(3.20, 1.938), xytext=(0.80, 1.938),
                arrowprops=dict(arrowstyle="-", lw=0.7, color=SUBTLE))
    ax.set_xticks(xs); ax.set_xticklabels([c["state"] for c in cyc])
    ax.set_ylabel(u"Co–O bond length (Å)")
    ax.set_xlim(-0.45, len(cyc) - 0.55); ax.set_ylim(1.50, 2.01)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    no_minor(ax, "x"); framed(ax)



def panel_cutoff_convergence(ax):
    x = [p[0] for p in ENCUT]; y = [p[1] for p in ENCUT]
    ax.axhspan(-1, 4, color=C_TERTIARY, alpha=0.10, lw=0)
    ax.plot(x, y, "-o", color=C_PRIMARY, mfc=C_PRIMARY, mec="white", mew=0.7)
    ax.plot([520], [3.613], "o", ms=7.5, mfc="none", mec=C_SECONDARY, mew=1.2)
    ax.annotate("520 eV", xy=(520, 3.613), xytext=(560, 5.3), fontsize=7,
                color=SUBTLE,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=C_SECONDARY))
    ax.set_xlabel("Plane-wave cutoff (eV)")
    ax.set_ylabel(r"$\Delta E$ vs 700 eV (meV atom$^{-1}$)")
    ax.set_xlim(375, 725); ax.set_ylim(-0.7, 6.8)
    framed(ax)


def panel_kpoint_convergence(ax):
    x = [p[0] for p in KSPACING]; y = [p[1] for p in KSPACING]
    ax.axhline(0, color=SUBTLE, lw=0.6)
    ax.plot(x, y, "-o", color=C_PRIMARY, mfc=C_PRIMARY, mec="white", mew=0.7)
    ax.plot([0.25], [-0.003], "o", ms=7.5, mfc="none", mec=C_SECONDARY, mew=1.2)
    ax.annotate(u"0.25 Å$^{-1}$", xy=(0.25, -0.003), xytext=(0.285, 0.085),
                fontsize=7, color=SUBTLE,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=C_SECONDARY))
    ax.invert_xaxis()
    ax.set_xlabel(u"$k$-point spacing (Å$^{-1}$)")
    ax.set_ylabel(r"$\Delta E$ vs finest (meV atom$^{-1}$)")
    ax.set_ylim(-0.035, 0.225)
    framed(ax)


def panel_spin_states(ax):
    xs = np.arange(len(SPIN_COOOH))
    v = [s[1] for s in SPIN_COOOH]
    ax.bar(xs, v, width=0.46, edgecolor="none", color=[C_PRIMARY, SUBTLE, SUBTLE])
    for x, y in zip(xs, v):
        ax.text(x, y + 0.016, f"{y:.3f}", ha="center", va="bottom", fontsize=7)
    ax.text(2, 0.30, "collapses\nto $S$ = 1", ha="center", va="center",
            fontsize=7, color=SUBTLE)
    ax.set_xticks(xs); ax.set_xticklabels([s[0] for s in SPIN_COOOH])
    ax.set_ylabel("Energy above ground state (eV)")
    ax.set_ylim(0, 0.75)
    no_minor(ax, "x"); framed(ax)


def panel_jahn_teller(ax):
    nm = [u"$P$1\n(Jahn–Teller)", u"$P\\bar{3}m$1\n(symmetrised)"]
    en, gaps = [0.0, 0.82], [2.60, 0.001]
    ax.bar([0, 1], en, width=0.42, edgecolor="none", color=[C_PRIMARY, C_SECONDARY])
    for x, (v, g) in enumerate(zip(en, gaps)):
        ax.text(x, v + 0.024, f"{v:.2f} eV", ha="center", va="bottom", fontsize=7)
        ax.text(x, 0.055, f"gap {g:.2f} eV" if g > 0.01 else "metallic",
                ha="center", va="bottom", fontsize=7,
                color="white" if v > 0.3 else SUBTLE)
    ax.set_xticks([0, 1]); ax.set_xticklabels(nm)
    ax.set_ylabel("Energy per formula unit (eV)")
    ax.set_ylim(0, 1.05)
    no_minor(ax, "x"); framed(ax)


def panel_oxo_solutions(ax):
    xs = np.arange(len(OXO))
    v = [o[1] for o in OXO]
    ax.axhline(0, color=SUBTLE, lw=0.6)
    ax.bar(xs, v, width=0.46, edgecolor="none",
           color=[C_SECONDARY, SUBTLE, C_PRIMARY])
    for x, (l, y, note) in enumerate(OXO):
        ax.text(x, y - 0.045, f"{y:+.3f}", ha="center", va="top", fontsize=7)
        ax.text(x, 0.075, note, ha="center", va="bottom", fontsize=7, color=SUBTLE)
    ax.text(-0.44, 0.30, r"moment on adsorbed O ($\mu_\mathrm{B}$)", fontsize=6.5,
            color=SUBTLE, ha="left")
    ax.set_xticks(xs); ax.set_xticklabels([o[0] for o in OXO])
    ax.set_ylabel("Energy vs unseeded solution (eV)")
    ax.set_ylim(-1.45, 0.55)
    no_minor(ax, "x"); framed(ax)


def panel_site_search(ax):
    xs = np.arange(len(SITES))
    v = [s[1] for s in SITES]
    ax.axhline(0, color=SUBTLE, lw=0.6)
    ax.bar(xs, v, width=0.46, edgecolor="none",
           color=[SUBTLE if y > 0 else C_PRIMARY for y in v])
    for x, y in zip(xs, v):
        ax.text(x, y + (0.07 if y > 0 else -0.07), f"{y:+.3f}", ha="center",
                va="bottom" if y > 0 else "top", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels([s[0] for s in SITES], fontsize=6.5)
    ax.set_ylabel("Energy vs assumed site (eV)")
    ax.set_ylim(-1.10, 2.05)
    no_minor(ax, "x"); framed(ax)



# ================================================ real electronic panels ====
def panel_phase_dos(ax):
    """Spin-resolved DOS across the reconstruction sequence: the gap closing
    is shown, not asserted with a bar."""
    ph = _phase_dos_series([r"$\beta$-Co(OH)$_2$", r"$\beta$-CoOOH",
                            "Co$_{0.5}$Ni$_{0.5}$OOH", r"$\beta$-NiOOH"])
    step = 1.0
    for j, (name, E, proj, gap) in enumerate(ph):
        base = -j * step
        w = (E > -6.5) & (E < 4.5)
        tot_u = np.zeros_like(E); tot_d = np.zeros_like(E)
        for el, (u, d) in proj.items():
            tot_u += u; tot_d += d
        nrm = max(tot_u[w].max(), tot_d[w].max(), 1e-9)
        for el, col in (("Co", C_PRIMARY), ("Ni", C_QUAT), ("O", C_SECONDARY)):
            if el not in proj:
                continue
            u, d = proj[el]
            ax.fill_between(E[w], base, base + 0.42 * u[w] / nrm, color=col,
                            alpha=0.5, lw=0)
            ax.fill_between(E[w], base, base - 0.42 * d[w] / nrm, color=col,
                            alpha=0.5, lw=0)
        ax.axhline(base, color=INK, lw=0.4, zorder=1)
        ax.text(-6.3, base + 0.30, name, fontsize=6.8, va="center")
        ax.text(4.35, base + 0.62, f"{gap:.2f} eV", fontsize=6.8,
                va="bottom", ha="right", color=SUBTLE)
    ax.axvline(0, color=INK, lw=0.7, ls=(0, (3, 2)), zorder=3)
    ax.text(0.16, 0.58, r"$E_\mathrm{F}$", fontsize=7.5, va="center")
    # colour key on its own row, clear of the first phase label
    for i, (lab, col) in enumerate((("Co 3$d$", C_PRIMARY), ("O 2$p$", C_SECONDARY),
                                    ("Ni 3$d$", C_QUAT))):
        ax.add_patch(plt.Rectangle((-6.30 + i * 1.85, 0.88), 0.30, 0.17,
                                   color=col, alpha=0.6, lw=0, clip_on=False))
        ax.text(-5.90 + i * 1.85, 0.965, lab, fontsize=6.5, va="center")
    ax.set_xlim(-6.5, 4.5); ax.set_ylim(-len(ph) * step + 0.45, 1.18)
    ax.set_yticks([])
    ax.set_xlabel(r"$E - E_\mathrm{F}$ (eV)")
    ax.set_ylabel("Projected DOS (arb. units)")
    no_minor(ax, "y"); framed(ax); ax.tick_params(axis="y", length=0)


def panel_active_site_dos(ax):
    """d-DOS on the ACTIVE cobalt through the four intermediates: the Co(IV)
    hole opening at the Fermi level, resolved by spin."""
    cyc = _cycle_dos()
    step = 1.0
    for j, (name, el, idx, E, u, d) in enumerate(cyc):
        base = -j * step
        w = (E > -4.0) & (E < 4.0)
        nrm = max(u[w].max(), d[w].max(), 1e-9)
        ax.fill_between(E[w], base, base + 0.42 * u[w] / nrm, color=C_PRIMARY,
                        alpha=0.55, lw=0)
        ax.fill_between(E[w], base, base - 0.42 * d[w] / nrm, color=C_SECONDARY,
                        alpha=0.55, lw=0)
        ax.axhline(base, color=INK, lw=0.4, zorder=1)
        ax.text(-3.85, base + 0.30, name, fontsize=7.2, va="center")
    ax.axvline(0, color=INK, lw=0.7, ls=(0, (3, 2)), zorder=3)
    ax.text(0.10, 0.52, r"$E_\mathrm{F}$", fontsize=7.5, va="center")
    ax.text(3.85, 0.30, "majority", fontsize=6.5, ha="right", color=INK)
    ax.text(3.85, -0.28, "minority", fontsize=6.5, ha="right", color=SUBTLE)
    ax.set_xlim(-4.0, 4.0); ax.set_ylim(-len(cyc) * step + 0.45, 0.62)
    ax.set_yticks([])
    ax.set_xlabel(r"$E - E_\mathrm{F}$ (eV)")
    ax.set_ylabel(r"Co 3$d$ DOS on the active site")
    no_minor(ax, "y"); framed(ax); ax.tick_params(axis="y", length=0)




def panel_covalency_map(ax):
    """Covalency map: O 2p band centre against band gap, not two bar charts."""
    seq = [(s[0], s[1], s[4]) for s in _sequence() if s[4] is not None]
    # hand-placed offsets: the three Ni-containing points cluster tightly
    OFF = {r"$\beta$-Co(OH)$_2$": (6, 10, "center"),
           r"$\beta$-CoOOH": (6, 10, "center"),
           "Co$_{0.75}$Ni$_{0.25}$OOH": (8, -14, "left"),
           "Co$_{0.5}$Ni$_{0.5}$OOH": (-8, 8, "right"),
           r"$\beta$-NiOOH": (-8, -12, "right")}
    order = sorted(seq, key=lambda t: t[2])
    ax.plot([s[2] for s in order], [s[1] for s in order], "-", color=SUBTLE,
            lw=0.7, zorder=1)
    for name, gap, o2p in seq:
        ni = "Ni" in name
        ax.plot(o2p, gap, "o", ms=6.5, mfc=C_TERTIARY if ni else C_PRIMARY,
                mec="white", mew=0.8, zorder=3)
        dx, dy, ha = OFF.get(name, (6, 10, "center"))
        ax.annotate(name, xy=(o2p, gap), xytext=(dx, dy),
                    textcoords="offset points", fontsize=6.5, ha=ha,
                    va="center")
    ax.annotate("", xy=(-2.62, 3.15), xytext=(-3.98, 3.15),
                arrowprops=dict(arrowstyle="->", lw=0.8, color=SUBTLE))
    ax.text(-3.30, 3.24, "more covalent", fontsize=6.5, ha="center", color=SUBTLE)
    ax.set_xlabel(r"O 2$p$ band center, $\varepsilon_{\mathrm{O}2p} - E_\mathrm{F}$ (eV)")
    ax.set_ylabel("Band gap (eV)")
    ax.set_xlim(-4.30, -2.35); ax.set_ylim(-0.35, 3.55)
    framed(ax)


# ================================================================ assembly ===
PANELS = {
    # main text
    "Fig16a_oer_diagram": panel_oer_diagram,
    "Fig16b_active_site_dos": panel_active_site_dos,
    "Fig16c_phase_dos": panel_phase_dos,
    "Fig16d_covalency_map": panel_covalency_map,
    # supplementary
    "FigS1a_hydrolysis": panel_hydrolysis,
    "FigS1b_lattice": panel_lattice_deviation,
    "FigS1c_mixing": panel_mixing_enthalpy,
    "FigS1d_linker_loss": panel_linker_loss,
    "FigS2a_hydrogen_binding": panel_hydrogen_binding,
    "FigS2b_water_dissociation": panel_water_dissociation,
    "FigS3a_redox_geometry": panel_redox_geometry,
    "FigS3b_pdos_pair": panel_pdos_pair,
    "FigS4a_cutoff": panel_cutoff_convergence,
    "FigS4b_kpoints": panel_kpoint_convergence,
    "FigS5a_oxo_solutions": panel_oxo_solutions,
    "FigS5b_site_search": panel_site_search,
}

COMPOSITES = [
    # ---- main text: one display item, four panels ------------------------
    ("Fig16_MAIN", [panel_oer_diagram, panel_active_site_dos,
                    panel_phase_dos, panel_covalency_map], 2, 2,
     (COL_DOUBLE, 5.2)),
    # ---- supplementary ---------------------------------------------------
    ("FigS1_reconstruction", [panel_hydrolysis, panel_lattice_deviation,
                              panel_mixing_enthalpy, panel_linker_loss], 2, 2,
     (COL_DOUBLE, 5.0)),
    ("FigS2_adsorption", [panel_hydrogen_binding, panel_water_dissociation],
     1, 2, (COL_DOUBLE, 2.6)),
    ("FigS3_electronic_detail", [panel_redox_geometry, panel_pdos_pair],
     1, 2, (COL_DOUBLE, 2.7)),
    ("FigS4_convergence", [panel_cutoff_convergence, panel_kpoint_convergence],
     1, 2, (COL_DOUBLE, 2.6)),
    ("FigS5_oxo_and_sites", [panel_oxo_solutions, panel_site_search],
     1, 2, (COL_DOUBLE, 2.6)),
]

SINGLE_SIZE = {  # panels needing more room when drawn standalone
    "Fig16a_oer_diagram":     (COL_SINGLE + 0.7, 2.8),
    "Fig16b_active_site_dos": (COL_SINGLE + 0.5, 3.0),
    "Fig16c_phase_dos":       (COL_SINGLE + 0.7, 3.0),
    "Fig16d_covalency_map":   (COL_SINGLE + 0.6, 2.7),
    "FigS1a_hydrolysis":      (COL_SINGLE + 0.7, 2.6),
    "FigS1b_lattice":         (COL_SINGLE + 0.6, 2.6),
    "FigS3b_pdos_pair":       (COL_SINGLE + 0.7, 2.8),
}


def build_composites():
    letters = "abcdefgh"
    for name, fns, nr, nc, size in COMPOSITES:
        fig, axes = plt.subplots(nr, nc, figsize=size, layout="constrained")
        axes = np.atleast_1d(axes).ravel()
        for i, fn in enumerate(fns):
            fn(axes[i])
            if len(fns) > 1:
                panel_label(axes[i], letters[i])
        for j in range(len(fns), len(axes)):
            axes[j].set_visible(False)
        save(fig, name, OUT_C)
        plt.close(fig)
        print(f"  composite/{name}")


def build_panels():
    for name, fn in PANELS.items():
        w, h = SINGLE_SIZE.get(name, (COL_SINGLE, 2.5))
        fig, ax = plt.subplots(figsize=(w, h), layout="constrained")
        fn(ax)
        save(fig, name, OUT_P)
        plt.close(fig)
    print(f"  panels/  {len(PANELS)} standalone figures")


def main() -> None:
    global ROOT, OUT_C, OUT_P
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True,
                        help="root of the calculation tree")
    parser.add_argument("--outdir", default="figures",
                        help="directory to write the figures into")
    args = parser.parse_args()

    ROOT = args.root
    OUT_C = os.path.join(args.outdir, "composite")
    OUT_P = os.path.join(args.outdir, "panels")

    load()
    build_composites()
    build_panels()
    print(f"Written to {OUT_C} and {OUT_P}")


if __name__ == "__main__":
    main()
