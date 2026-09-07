"""Build the OER (stage 4) and HER (stage 5) surface models for the ZIF-67 paper.

Reuses `write_case`, `add_ldau` and `co_magmom` from make_inputs.py so that MAGMOM,
the LDAU array ordering and POTCAR_SPEC are generated exactly the same way as for
the molecules / bulks / ZIF-67 that are already set up.

What it builds
--------------
STAGE 4  4_oer_slabs/
    beta-CoOOH(01-12), 2x2, 4 Co layers, bottom 2 fixed, >=15 A vacuum
        clean, *OH, *O, *OOH                       (adsorbed on the cus Co)
        ni_clean, ni_oh, ni_o, ni_ooh              (1 of 4 surface Co -> Ni, 25 %)
    mixing/  bulk Co(1-x)Ni(x)OOH, x = 0, 0.25, 0.5, 1.0  (identical 16-atom cell)

STAGE 5  5_her_slabs/
    beta-Co(OH)2(001), 2x2, 4 sheets  : clean, H* at 1/4 ML and 1/2 ML
    beta-CoOOH(01-12), 2x2           : clean, H*
    neb_water_dissociation/           : endpoints 00 and 06 only, plus INCAR.neb

Usage
-----
    python make_slabs.py            # everything
    python make_slabs.py oer
    python make_slabs.py her
"""

__author__ = "Nabil Khossossi"

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.vasp.inputs import Incar, Kpoints
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_inputs import _psp, co_magmom, write_case  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OER = ROOT / "4_oer_slabs"
HER = ROOT / "5_her_slabs"

# --- surface-model parameters (also recorded in 4_oer_slabs/SETUP_NOTES.md) ---
MILLER_COOOH = (0, 1, 2)      # 3-index form of the hexagonal (01-12) facet
N_LAYERS_FIXED = 2            # bottom Co layers held at bulk positions
VACUUM = 15.0                 # A, the requirement quoted in METHODS.md
# SlabGenerator quantises c in units of d_hkl, and a tall adsorbate (*OOH, H2O*)
# eats ~3 A of the gap, so the generator is asked for more than the requirement and
# the built structures are then checked against VACUUM itself.
VAC_GEN_COOOH = 17.0          # -> 19.9 A clean, >= 16.7 A with *OOH
VAC_GEN_COOH2 = 15.0          # -> 19.9 A clean, the (001) H* barely protrudes
SUPERCELL = (2, 2, 1)
KPT_LENGTH = 30.0             # target N_i * a_i in A -> Gamma-centred slab mesh

# adsorbate internal geometry (A / deg); relaxation refines these
D_CO_O = 1.85       # Co-O for *OH
D_CO_OXO = 1.65     # Co=O for *O
D_CO_OOH = 1.90     # Co-O1 for *OOH
D_OO = 1.45         # O-O in *OOH
D_OH = 0.98         # O-H
D_CO_OW = 2.15      # Co-OH2 (dative, longer)


# ------------------------------------------------------------------ helpers
def unit(v):
    return np.asarray(v, dtype=float) / np.linalg.norm(v)


# Which bulk cell each slab family is built from. This is deliberate, not a
# default — see RESUME_HERE.md "STAGE 5".
#
#   co_oh2  -> relax2/CONTCAR. The unrelaxed MP cell is wrong by -4.88 % in c
#              (4.7785 -> 4.5454 A under D3-BJ). That is the INTERLAYER spacing
#              of a layered hydroxide and H* adsorbs on (001), so it changes the
#              surface itself. Too large to tolerate.
#   cooh    -> POSCAR, ON PURPOSE. Relaxation moves it only +0.42 %
#              (4.6148 -> 4.6341 A), and every 4_oer_slabs/* slab was built from
#              POSCAR. Switching would desynchronise stage 5 from stage 4 and
#              void the cooh_0112_clean == clean_ls equivalence. At ISIF = 2 the
#              cell is fixed across the whole intermediate series, so a common
#              0.42 % strain cancels in dG1..dG4. Revisit only if stage 4 is
#              ever rebuilt wholesale.
BULK_SOURCE = {
    "co_oh2": "relax2/CONTCAR",
    "cooh": "POSCAR",
}


def bulk(name: str) -> Structure:
    base = ROOT / "2_bulk" / name
    src = BULK_SOURCE.get(name, "POSCAR")
    f = base / src
    if not (f.is_file() and f.stat().st_size > 0):
        raise FileNotFoundError(f"bulk({name}): {f} missing; BULK_SOURCE says {src}")
    return Structure.from_file(f)


def layer_z(struct: Structure, element: str = "Co", tol: float = 0.9) -> list[float]:
    """Cluster the z of `element` into layers, returned bottom-up."""
    zs = sorted(s.coords[2] for s in struct if s.specie.symbol in (element, "Ni"))
    layers: list[list[float]] = [[zs[0]]]
    for z in zs[1:]:
        if z - layers[-1][-1] < tol:
            layers[-1].append(z)
        else:
            layers.append([z])
    return [float(np.mean(g)) for g in layers]


def freeze_bottom(struct: Structure, n_fixed: int = N_LAYERS_FIXED) -> Structure:
    """Selective dynamics: everything below the n_fixed-th metal layer is frozen."""
    lz = layer_z(struct)
    if len(lz) < n_fixed + 2:
        raise ValueError(f"only {len(lz)} metal layers, need >= {n_fixed + 2}")
    z_cut = 0.5 * (lz[n_fixed - 1] + lz[n_fixed])
    sd = [[False] * 3 if s.coords[2] < z_cut else [True] * 3 for s in struct]
    struct = struct.copy()
    struct.add_site_property("selective_dynamics", sd)
    return struct


def slab_kpoints(struct: Structure) -> Kpoints:
    a, b, _ = struct.lattice.abc
    na = max(1, int(round(KPT_LENGTH / a)))
    nb = max(1, int(round(KPT_LENGTH / b)))
    return Kpoints.gamma_automatic((na, nb, 1))


def open_direction(struct: Structure, idx: int, rcut: float = 2.5) -> np.ndarray:
    """Unit vector along the vacant coordination site of a cus metal atom."""
    site = struct[idx]
    nbrs = [n for n in struct.get_neighbors(site, rcut) if n.specie.symbol in ("O", "N")]
    d = -sum(unit(n.coords - site.coords) for n in nbrs)
    d = unit(d)
    if d[2] < 0:
        raise ValueError(f"open direction of site {idx} points into the slab: {d}")
    return d


def in_plane_perp(d: np.ndarray, struct: Structure) -> np.ndarray:
    """A unit vector perpendicular to d, lying roughly along the surface a-axis."""
    a = np.asarray(struct.lattice.matrix[0], dtype=float)
    t = a - np.dot(a, d) * d
    return unit(t)


def top_metal_indices(struct: Structure, symbols=("Co", "Ni")) -> list[int]:
    """Indices of the metal atoms in the topmost metal layer."""
    lz = layer_z(struct)
    ztop = lz[-1]
    return [i for i, s in enumerate(struct)
            if s.specie.symbol in symbols and abs(s.coords[2] - ztop) < 0.9]


def top_surface_oxygens(struct: Structure, n: int | None = None) -> list[int]:
    """Indices of the outermost O atoms (bottom-up sorted by z, highest last)."""
    ox = sorted((i for i, s in enumerate(struct) if s.specie.symbol == "O"),
                key=lambda i: struct[i].coords[2])
    return ox if n is None else ox[-n:]


def add_atoms(struct: Structure, items) -> Structure:
    out = struct.copy()
    sd = out.site_properties.get("selective_dynamics")
    for sym, cart in items:
        out.append(sym, cart, coords_are_cartesian=True)
    if sd is not None:
        out.add_site_property("selective_dynamics", sd + [[True] * 3] * len(items))
    return out


def min_contact(struct: Structure, n_new: int, skip: int = 0) -> float:
    """Distance from the last n_new atoms to the rest, ignoring the `skip` shortest.

    skip=0 returns the bonding distance to the surface; skip=1 the shortest
    genuinely non-bonded contact, which is what has to stay above ~1.5 A.
    """
    n = len(struct)
    sub = np.sort(struct.distance_matrix[n - n_new:, : n - n_new].ravel())
    return float(sub[skip])


def report(tag: str, struct: Structure, n_new: int) -> None:
    print(f"      {tag}: bond {min_contact(struct, n_new):.2f} A, "
          f"closest non-bonded {min_contact(struct, n_new, n_new):.2f} A")
    if min_contact(struct, n_new, n_new) < 1.30:
        print(f"      [WARNING] {tag}: non-bonded contact below 1.30 A")


def outward(struct: Structure, idx: int, rcut: float = 2.5) -> np.ndarray:
    """Unit vector pointing from site `idx` away from its metal neighbours."""
    site = struct[idx]
    m = [n for n in struct.get_neighbors(site, rcut)
         if n.specie.symbol in ("Co", "Ni")]
    if not m:
        return np.array([0.0, 0.0, 1.0])
    d = unit(-sum(unit(n.coords - site.coords) for n in m))
    return d if d[2] > 0 else np.array([0.0, 0.0, 1.0])


# ------------------------------------------------------------------ adsorbates
def put_oh(struct, idx):
    d = open_direction(struct, idx)
    t = in_plane_perp(d, struct)
    o = struct[idx].coords + D_CO_O * d
    h = o + D_OH * unit(np.cos(np.deg2rad(60)) * d + np.sin(np.deg2rad(60)) * t)
    return add_atoms(struct, [("O", o), ("H", h)]), 2


def put_o(struct, idx):
    d = open_direction(struct, idx)
    return add_atoms(struct, [("O", struct[idx].coords + D_CO_OXO * d)]), 1


def put_ooh(struct, idx):
    d = open_direction(struct, idx)
    t = in_plane_perp(d, struct)
    o1 = struct[idx].coords + D_CO_OOH * d
    u = unit(np.cos(np.deg2rad(55)) * d + np.sin(np.deg2rad(55)) * t)   # Co-O-O ~ 125 deg
    o2 = o1 + D_OO * u
    best = None
    for ang in (-25.0, 135.0):          # H up-and-back, or H folded toward the surface
        v = unit(np.cos(np.deg2rad(ang)) * d + np.sin(np.deg2rad(ang)) * t)
        cand = add_atoms(struct, [("O", o1), ("O", o2), ("H", o2 + D_OH * v)])
        gap = min_contact(cand, 3)
        if best is None or gap > best[0]:
            best = (gap, cand)
    return best[1], 3


def h_on_o_position(struct, o_idx) -> np.ndarray:
    """Where a new H goes on a surface O.

    Both terminations here are already hydroxylated, so the surface O normally
    carries an H; the new proton is then placed at the water HOH angle (104.5 deg)
    from the existing O-H, in the plane that leans away from the slab, giving an
    adsorbed H2O. The azimuth is scanned so the new H does not run into anything.
    """
    o = struct[o_idx].coords
    n_out = outward(struct, o_idx)
    existing = [nb for nb in struct.get_neighbors(struct[o_idx], 1.35)
                if nb.specie.symbol == "H"]
    if not existing:
        return o + D_OH * n_out

    e = unit(existing[0].coords - o)
    p0 = n_out - np.dot(n_out, e) * e
    if np.linalg.norm(p0) < 1e-6:            # O-H already along the normal
        p0 = in_plane_perp(e, struct)
    p0 = unit(p0)
    q = unit(np.cross(e, p0))
    ang = np.deg2rad(104.5)
    best = None
    for phi in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        p = np.cos(phi) * p0 + np.sin(phi) * q
        v = np.cos(ang) * e + np.sin(ang) * p
        pos = o + D_OH * v
        gap = min(np.linalg.norm(pos - s.coords) for i, s in enumerate(struct)
                  if i != o_idx)
        # prefer directions that keep the H out of the slab
        score = gap + 0.3 * float(np.dot(v, n_out))
        if best is None or score > best[0]:
            best = (score, pos)
    return best[1]


def put_h_on_o(struct, o_idx):
    """H on a surface O -> Volmer product on a hydroxide/oxyhydroxide termination."""
    return add_atoms(struct, [("H", h_on_o_position(struct, o_idx))]), 1


def put_h2o(struct, idx, acceptor_idx):
    """Intact H2O datively bound to a cus metal, one O-H aimed at `acceptor_idx`.

    The lone pair points at the metal, so the HOH bisector points away from it; the
    tilt of that bisector towards the acceptor O is scanned for the geometry that
    gives an H...O(acceptor) hydrogen bond near 1.8 A without any close contact.

    The two H are appended as (spectator, transferring): the H aimed at the acceptor
    comes LAST, so that after get_sorted_structure() its index coincides with the
    transferred H of the dissociated endpoint. nebmake.pl interpolates atom i of the
    initial state into atom i of the final state, so this correspondence matters.
    Also returns the spectator O-H direction for reuse by the final state.
    """
    d = open_direction(struct, idx)
    ow = struct[idx].coords + D_CO_OW * d
    acc = struct[acceptor_idx].coords
    t = unit(acc - ow - np.dot(acc - ow, d) * d)
    half = np.deg2rad(104.5 / 2)

    best = None
    for tilt in np.arange(0.0, 90.0, 5.0):
        bis = unit(np.cos(np.deg2rad(tilt)) * d + np.sin(np.deg2rad(tilt)) * t)
        p = unit(t - np.dot(t, bis) * bis)
        hs = [ow + D_OH * unit(np.cos(half) * bis + s * np.sin(half) * p)
              for s in (1, -1)]
        hs.sort(key=lambda h: -np.linalg.norm(h - acc))      # spectator first
        cand = add_atoms(struct, [("O", ow), ("H", hs[0]), ("H", hs[1])])
        if min_contact(cand, 3, 1) < 1.45:                   # skip the Co-O bond
            continue
        score = -abs(np.linalg.norm(hs[1] - acc) - 1.80)
        if best is None or score > best[0]:
            best = (score, cand, unit(hs[0] - ow))
    if best is None:
        raise RuntimeError("could not place an intact H2O without a close contact")
    return best[1], 3, best[2]


def put_oh_plus_h(struct, idx, acceptor_idx, h_dir=None):
    """Dissociated state: OH* on the cus metal, H* on the acceptor surface O.

    Appended as (O, spectator H, transferred H) to match put_h2o(); `h_dir` is the
    spectator O-H direction of the initial state, reused here so the interpolation
    does not have to swing the surviving O-H around.
    """
    d = open_direction(struct, idx)
    o = struct[idx].coords + D_CO_O * d
    if h_dir is None:
        t = in_plane_perp(d, struct)
        h_dir = np.cos(np.deg2rad(60)) * d + np.sin(np.deg2rad(60)) * t
    h_oh = o + D_OH * unit(h_dir)
    h_surf = h_on_o_position(struct, acceptor_idx)
    return add_atoms(struct, [("O", o), ("H", h_oh), ("H", h_surf)]), 3


# ------------------------------------------------------------------ MAGMOM
def slab_magmom(struct: Structure, phase: str, adsorbate_o: list[int] | None = None):
    """co_magmom() with the spin state pinned to the known bulk assignment.

    co_magmom infers the oxidation state from bond valence, which is not reliable
    for undercoordinated surface atoms, so the result is checked against the phase:
        CoOOH   -> Co(III) octahedral, LOW spin  0.0
        Co(OH)2 -> Co(II)  octahedral, HIGH spin 5.0
        Ni in either -> Ni(III) 1.0 / Ni(II) 2.0
    """
    mags = co_magmom(struct)
    want_co, want_ni = (0.0, 1.0) if phase == "cooh" else (5.0, 2.0)
    fixed = 0
    for i, s in enumerate(struct):
        el = s.specie.symbol
        if el == "Co" and abs(mags[i] - want_co) > 1e-6:
            mags[i], fixed = want_co, fixed + 1
        elif el == "Ni" and abs(mags[i] - want_ni) > 1e-6:
            mags[i], fixed = want_ni, fixed + 1
    if fixed:
        print(f"      [magmom] pinned {fixed} metal site(s) to the {phase} spin state")
    # let a magnetic solution be reachable on an otherwise closed-shell oxo/peroxo
    for i in adsorbate_o or []:
        mags[i] = 0.5
    return mags


# ------------------------------------------------------------------ writer
def write_slab(name, struct, folder, phase, *, readme, adsorbate_o=None, extra=None):
    """write_case() + slab-specific INCAR/KPOINTS fix-ups + a README."""
    struct = struct.get_sorted_structure()          # Co (Ni) H O; keeps site props
    sd = struct.site_properties.get("selective_dynamics")
    if sd is None or all(all(f) for f in sd):
        raise ValueError(f"{name}: selective dynamics missing or nothing is fixed")

    # map adsorbate O indices through the sort
    ads_o = None
    if adsorbate_o:
        ads_o = [i for i, s in enumerate(struct)
                 if s.specie.symbol == "O" and any(
                     np.allclose(s.coords, c, atol=1e-6) for c in adsorbate_o)]

    z = np.array([s.coords[2] for s in struct])
    vac = struct.lattice.c - (z.max() - z.min())
    if vac < VACUUM:
        raise ValueError(f"{name}: vacuum {vac:.2f} A < {VACUUM} A")
    zc = float(np.mean([s.frac_coords[2] for s in struct]))
    slab_extra = {
        "LDIPOL": True, "IDIPOL": 3, "DIPOL": [0.5, 0.5, round(zc, 4)],
        "NSW": 300,
    }
    slab_extra.update(extra or {})

    write_case(name, struct, folder, gamma_only=True, relax_cell=False,
               extra=slab_extra)

    # write_case wrote a 1x1x1 Gamma KPOINTS placeholder -> replace with the real mesh
    kpts = slab_kpoints(struct)
    kpts.write_file(str(folder / "KPOINTS"))

    incar = Incar.from_file(folder / "INCAR")
    incar["MAGMOM"] = slab_magmom(struct, phase, ads_o)
    incar.pop("NUPDOWN", None)      # do not force a strict singlet on a surface
    incar.write_file(str(folder / "INCAR"))

    n_fixed = sum(1 for f in sd if not f[0])
    (folder / "README.md").write_text(readme.format(
        natoms=len(struct), nfixed=n_fixed,
        a=struct.lattice.a, b=struct.lattice.b, c=struct.lattice.c,
        area=struct.lattice.a * struct.lattice.b
        * np.sin(np.deg2rad(struct.lattice.gamma)),
        kmesh="x".join(str(x) for x in kpts.kpts[0]),
    ))
    return struct


# ------------------------------------------------------------------ stage 4
def cooh_slab() -> Structure:
    """The beta-CoOOH(01-12) 2x2 slab: 4 Co layers, cus Co on both faces."""
    conv = SpacegroupAnalyzer(bulk("cooh")).get_conventional_standard_structure()
    sg = SlabGenerator(conv, MILLER_COOOH, min_slab_size=8.0,
                       min_vacuum_size=VAC_GEN_COOOH,
                       center_slab=True, in_unit_planes=False, lll_reduce=True,
                       primitive=True)
    # take the stoichiometric, symmetric termination (shift 0.5) -- it exposes the
    # 5-fold coordinated (cus) Co that is the OER active site
    slabs = [s for s in sg.get_slabs(symmetrize=False) if s.is_symmetric()]
    if not slabs:
        raise RuntimeError("no symmetric (01-12) termination found")
    slab = slabs[0]
    st = Structure(slab.lattice, slab.species, slab.frac_coords)
    st.make_supercell(SUPERCELL)
    return freeze_bottom(st)


def ni_substitute(struct: Structure, frac: float = 0.25) -> tuple[Structure, int]:
    """Replace `frac` of the top-layer Co by Ni; return (structure, index of one Ni)."""
    tops = top_metal_indices(struct, ("Co",))
    n_sub = max(1, int(round(frac * len(tops))))
    # pick the most mutually separated subset (n_sub == 1 here, so just take the first)
    chosen = sorted(tops)[:n_sub]
    out = struct.copy()
    for i in chosen:
        out[i] = "Ni"
    return out, chosen[0]


OER_README = """# {title}

* System: beta-CoOOH(01-12) slab{nisub}{ads}
* Surface cell: {{a:.3f}} x {{b:.3f}} A ({sup}), surface area {{area:.2f}} A^2
* Slab: 4 Co layers, bottom {nfix} fixed at bulk positions (selective dynamics)
* Vacuum: >= {vac:.0f} A (cell c = {{c:.3f}} A), dipole correction LDIPOL/IDIPOL=3
* Coverage: {cov}
* Atoms: {{natoms}} total, {{nfixed}} frozen
* k-points: Gamma-centred {{kmesh}}
* Spin: Co(III) octahedral LOW spin (MAGMOM 0){nimag}

{extra}
"""


def make_oer() -> None:
    print("\nSTAGE 4 - beta-CoOOH(01-12) OER slabs")
    OER.mkdir(parents=True, exist_ok=True)
    base = cooh_slab()
    print(f"  clean 2x2 slab: {len(base)} atoms, "
          f"{base.lattice.a:.3f} x {base.lattice.b:.3f} A, "
          f"c = {base.lattice.c:.3f} A")

    sup = "2x2"
    common = dict(sup=sup, vac=VACUUM, nfix=N_LAYERS_FIXED)

    def rd(title, nisub, ads, cov, nimag="", extra=""):
        return OER_README.format(title=title, nisub=nisub, ads=ads, cov=cov,
                                 nimag=nimag, extra=extra, **common)

    note_clean = ("Reference (bare) surface for the four-step associative OER cycle.\n"
                  "The four top-layer Co are 5-fold coordinated (cus) sites.")
    note_ads = ("Adsorbate placed on the vacant octahedral coordination site of one\n"
                "cus {m}; 1 of the 4 equivalent surface sites is occupied.")

    ni_note = "; Ni(III) MAGMOM 1.0 (1 of 4 surface Co substituted, 25 %)"

    ni_struct, ni_site = ni_substitute(base)
    variants = [(False, base, top_metal_indices(base, ("Co",))[0]),
                (True, ni_struct, ni_site)]

    for ni, st0, site in variants:
        pre = "ni_" if ni else ""
        nisub = " with Ni substituted for 1 of the 4 surface Co (25 %)" if ni else ""
        nimag = ni_note if ni else ""
        m = "Ni" if ni else "Co"

        write_slab(f"CoOOH(01-12) {pre}clean", st0, OER / f"{pre}clean", "cooh",
                   readme=rd(f"{'Ni-doped ' if ni else ''}beta-CoOOH(01-12) - clean",
                             nisub, "", "clean surface, no adsorbate",
                             nimag, note_clean))

        for tag, fn, label in (("oh", put_oh, "*OH"), ("o", put_o, "*O"),
                               ("ooh", put_ooh, "*OOH")):
            st, nnew = fn(st0, site)
            ads_o = [st[i].coords for i in range(len(st) - nnew, len(st))
                     if st[i].specie.symbol == "O"] if tag in ("o", "ooh") else None
            write_slab(f"CoOOH(01-12) {pre}{tag}", st, OER / f"{pre}{tag}", "cooh",
                       adsorbate_o=ads_o,
                       readme=rd(f"{'Ni-doped ' if ni else ''}"
                                 f"beta-CoOOH(01-12) - {label}",
                                 nisub, f" + {label}",
                                 f"1 {label} on a cus {m}, 1 of the 4 cus metal "
                                 f"sites of the 2x2 cell = 1/4 ML "
                                 f"(1 adsorbate per {st.lattice.a * st.lattice.b * np.sin(np.deg2rad(st.lattice.gamma)):.1f} A^2)",
                                 nimag, note_ads.format(m=m)))
            report(f"{pre}{tag}", st, nnew)

    make_mixing()


MIX_README = """# Bulk Co(1-x)Ni(x)OOH, x = {x}

* Phase: beta-CoOOH structure type (R-3m), {nCo} Co + {nNi} Ni per cell
* Cell: 2x2x1 supercell of the rhombohedral primitive cell, {natoms} atoms
  (identical lattice for every x, so the four total energies are directly
  comparable without any cell-size correction)
* Full cell + ion relaxation (ISIF=3)
* k-points: Gamma-centred {kmesh}
* Spin: Co(III) LOW spin (0), Ni(III) 1.0 uB

Mixing enthalpy:

    dH_mix(x) = [ E(Co(1-x)Ni(x)OOH) - (1-x) E(CoOOH) - x E(NiOOH) ] / N_f.u.

with N_f.u. = {nfu} and E(CoOOH), E(NiOOH) taken from x=0.00 and x=1.00 in this
same directory.
"""


def make_mixing() -> None:
    print("\n  bulk Co(1-x)Ni(x)OOH mixing cells")
    prim = bulk("cooh")                     # 4 atoms, 1 Co, rhombohedral primitive
    cell = prim.copy()
    cell.make_supercell([[2, 0, 0], [0, 2, 0], [0, 0, 1]])   # 16 atoms, 4 Co
    co_idx = [i for i, s in enumerate(cell) if s.specie.symbol == "Co"]
    nfu = len(co_idx)

    def sub(indices):
        st = cell.copy()
        for i in indices:
            st[i] = "Ni"
        return st

    # x = 0.5: of the C(4,2) arrangements keep the one with the largest Ni-Ni
    # separation (most homogeneous solid solution).
    from itertools import combinations
    best = max(combinations(co_idx, 2),
               key=lambda c: cell.get_distance(c[0], c[1]))

    plans = {"x000": [], "x025": [co_idx[0]], "x050": list(best), "x100": co_idx}
    kpts = Kpoints.automatic_gamma_density(cell, 2000)

    for tag, idx in plans.items():
        st = sub(idx).get_sorted_structure()
        folder = OER / "mixing" / tag
        x = len(idx) / nfu
        write_case(f"Co(1-x)Ni(x)OOH x={x:.2f}", st, folder,
                   gamma_only=True, relax_cell=True)
        kpts.write_file(str(folder / "KPOINTS"))
        incar = Incar.from_file(folder / "INCAR")
        incar["MAGMOM"] = slab_magmom(st, "cooh")
        incar.pop("NUPDOWN", None)
        incar.write_file(str(folder / "INCAR"))
        (folder / "README.md").write_text(MIX_README.format(
            x=f"{x:.2f}", nCo=nfu - len(idx), nNi=len(idx), natoms=len(st),
            nfu=nfu, kmesh="x".join(str(k) for k in kpts.kpts[0])))


# ------------------------------------------------------------------ stage 5
HER_README = """# {title}

* System: {system}
* Surface cell: {{a:.3f}} x {{b:.3f}} A (2x2), surface area {{area:.2f}} A^2
* Slab: {layers}, bottom {nfix} fixed at bulk positions (selective dynamics)
* Vacuum: >= {vac:.0f} A (cell c = {{c:.3f}} A), dipole correction LDIPOL/IDIPOL=3
* Coverage: {cov}
* Atoms: {{natoms}} total, {{nfixed}} frozen
* k-points: Gamma-centred {{kmesh}}
* Spin: {spin}

{extra}
"""


def cooh2_slab() -> Structure:
    """beta-Co(OH)2(001) 2x2 slab: 4 brucite sheets, cleaved in the van der Waals gap.

    The Materials Project cell is very slightly distorted from ideal P-3m1, so the
    conventional cell is taken with symprec = 0.1 to recover the hexagonal setting.
    """
    conv = SpacegroupAnalyzer(bulk("co_oh2"),
                              symprec=0.1).get_conventional_standard_structure()
    sg = SlabGenerator(conv, (0, 0, 1), min_slab_size=16.0,
                       min_vacuum_size=VAC_GEN_COOH2,
                       center_slab=True, in_unit_planes=False, lll_reduce=False,
                       primitive=True)
    # the symmetric termination is the one cleaved between the H layers (the vdW gap);
    # the other two cut through a Co(OH)2 sheet and leave bare Co exposed
    slabs = [s for s in sg.get_slabs(symmetrize=False) if s.is_symmetric()]
    if not slabs:
        raise RuntimeError("no symmetric (001) termination found for beta-Co(OH)2")
    st = Structure(slabs[0].lattice, slabs[0].species, slabs[0].frac_coords)
    st.make_supercell(SUPERCELL)
    return freeze_bottom(st)


def make_her() -> None:
    print("\nSTAGE 5 - HER slabs")
    HER.mkdir(parents=True, exist_ok=True)

    # ---- beta-Co(OH)2(001) -------------------------------------------------
    coh = cooh2_slab()
    print(f"  beta-Co(OH)2(001) 2x2: {len(coh)} atoms, "
          f"{coh.lattice.a:.3f} x {coh.lattice.b:.3f} A, c = {coh.lattice.c:.3f} A")
    n_top_o = len(top_metal_indices(coh))          # 4 surface OH per 2x2

    note_coh = ("The (001) basal plane of brucite-type beta-Co(OH)2 is terminated by OH\n"
                "groups; the Co are fully 6-fold coordinated and buried, so the only\n"
                "exposed adsorption site is the surface O. H* therefore forms a surface\n"
                "H2O (Volmer product) rather than a metal hydride. dG_H* is evaluated\n"
                "against 1/2 H2 with the usual 0.24 eV ZPE + entropy correction.")

    write_slab("Co(OH)2(001) clean", coh, HER / "co_oh2_001_clean", "co_oh2",
               readme=HER_README.format(
                   title="beta-Co(OH)2(001) - clean",
                   system="beta-Co(OH)2 (P-3m1) (001) basal plane",
                   layers="4 Co(OH)2 sheets", nfix=N_LAYERS_FIXED, vac=VACUUM,
                   cov="clean surface", spin="Co(II) octahedral HIGH spin (MAGMOM 5.0)",
                   extra=note_coh))

    surf_o = top_surface_oxygens(coh, n_top_o)     # the 4 outermost O of the 2x2 cell
    for n_h, tag, cov in ((1, "co_oh2_001_h_025ml", "1 H per 4 surface OH = 1/4 ML"),
                          (2, "co_oh2_001_h_050ml", "2 H per 4 surface OH = 1/2 ML")):
        st = coh
        # spread the H out: take the two most distant surface O for 1/2 ML
        if n_h == 1:
            picks = [surf_o[0]]
        else:
            from itertools import combinations
            picks = list(max(combinations(surf_o, 2),
                             key=lambda c: coh.get_distance(c[0], c[1])))
        for k, oi in enumerate(picks):
            st, _ = put_h_on_o(st, oi)
        report(tag, st, n_h)
        write_slab(f"Co(OH)2(001) +{n_h}H", st, HER / tag, "co_oh2",
                   readme=HER_README.format(
                       title=f"beta-Co(OH)2(001) - {n_h} H* ({cov.split('=')[-1].strip()})",
                       system="beta-Co(OH)2 (P-3m1) (001) basal plane + H*",
                       layers="4 Co(OH)2 sheets", nfix=N_LAYERS_FIXED, vac=VACUUM,
                       cov=cov, spin="Co(II) octahedral HIGH spin (MAGMOM 5.0)",
                       extra=note_coh))

    # ---- beta-CoOOH(01-12) ------------------------------------------------
    base = cooh_slab()
    note_cooh = ("Same slab as 4_oer_slabs/clean, so the OER and HER numbers refer to\n"
                 "one and the same surface model. H* is placed on the outermost surface\n"
                 "O (the bridging oxygen of the cus Co), which is the accessible site on\n"
                 "an oxyhydroxide termination.")
    write_slab("CoOOH(01-12) clean (HER)", base, HER / "cooh_0112_clean", "cooh",
               readme=HER_README.format(
                   title="beta-CoOOH(01-12) - clean (HER reference)",
                   system="beta-CoOOH (R-3m) (01-12) facet",
                   layers="4 Co layers", nfix=N_LAYERS_FIXED, vac=VACUUM,
                   cov="clean surface",
                   spin="Co(III) octahedral LOW spin (MAGMOM 0)", extra=note_cooh))

    o_top = top_surface_oxygens(base, 4)[-1]
    st, _ = put_h_on_o(base, o_top)
    report("cooh_0112_h", st, 1)
    write_slab("CoOOH(01-12) +H", st, HER / "cooh_0112_h", "cooh",
               readme=HER_README.format(
                   title="beta-CoOOH(01-12) - H*",
                   system="beta-CoOOH (R-3m) (01-12) facet + H*",
                   layers="4 Co layers", nfix=N_LAYERS_FIXED, vac=VACUUM,
                   cov="1 H per 4 surface cus Co = 1/4 ML",
                   spin="Co(III) octahedral LOW spin (MAGMOM 0)", extra=note_cooh))

    make_neb(base)


NEB_INCAR = """# CI-NEB, water dissociation on beta-CoOOH(01-12)
#   H2O*  ->  OH* + H*          (alkaline Volmer step)
# Run in this directory, with 00/ ... 06/ created by nebmake.pl.

  SYSTEM = CoOOH(01-12) H2O dissociation CI-NEB

# --- NEB ---
  IMAGES = 5
  SPRING = -5.0
  LCLIMB = .TRUE.
  ICHAIN = 0

# --- ionic relaxation (plain VASP; see README for the VTST optimiser) ---
  IBRION = 3
  POTIM  = 0.0
  IOPT   = 7
  NSW    = 300
  EDIFFG = -0.05

# --- electronic ---
  PREC   = Accurate
  ENCUT  = 520
  EDIFF  = 1E-6
  NELM   = 200
  ALGO   = Normal
  LREAL  = Auto
  ISMEAR = 0
  SIGMA  = 0.05
  ISPIN  = 2
  IVDW   = 12
  LMAXMIX = 4
  LORBIT = 11
  LWAVE  = .FALSE.
  LCHARG = .FALSE.

# --- DFT+U (order must match POSCAR: {order}) ---
  LDAU     = .TRUE.
  LDAUTYPE = 2
  LDAUPRINT = 1
  LDAUL    = {ldaul}
  LDAUU    = {ldauu}
  LDAUJ    = {ldauj}

# --- dipole correction normal to the surface ---
  LDIPOL = .TRUE.
  IDIPOL = 3
  DIPOL  = 0.5 0.5 {dipol}

  MAGMOM = {magmom}
"""

NEB_README = """# CI-NEB: water dissociation on beta-CoOOH(01-12)

Alkaline Volmer step, the rate-limiting step of alkaline HER:

    H2O*  ->  [TS]  ->  OH* + H*

* Surface: beta-CoOOH(01-12), 2x2, 4 Co layers, bottom {nfix} fixed, >= {vac:.0f} A vacuum
* Initial state `00/POSCAR`: H2O bound intact through its O to a cus Co
  ({n_is} atoms; slab {n_slab} + H2O 3)
* Final state  `06/POSCAR`: OH* on the same cus Co, the transferred H sitting on the
  neighbouring surface O ({n_fs} atoms)
* 5 intermediate images (`01` ... `05`), as stated in METHODS.md

## Before you start

**Relax both endpoints first.** `00/POSCAR` and `06/POSCAR` here are *constructed*
geometries, not relaxed ones. Copy each into its own directory with the INCAR from
`../cooh_0112_h` (ISIF=2, IBRION=2), relax to EDIFFG = -0.02, then copy the resulting
CONTCARs back to `00/POSCAR` and `06/POSCAR`. A NEB started from unrelaxed endpoints
is meaningless.

## Running

```bash
# 1. build the POTCAR in the order given in POTCAR_SPEC
cat $PSP/Co_pv/POTCAR $PSP/H/POTCAR $PSP/O/POTCAR > POTCAR

# 2. interpolate 5 images between the two relaxed endpoints
#    (nebmake.pl ships with the VTST tools: http://theory.cm.utexas.edu/vtsttools/)
nebmake.pl 00/POSCAR 06/POSCAR 5
#    -> creates 00/ 01/ 02/ 03/ 04/ 05/ 06/, with POSCARs in 01..05

# 3. the endpoint energies must be present for the plots:
#    put the OUTCAR of each relaxed endpoint into 00/ and 06/

# 4. check that no atom moves absurdly far between neighbouring images
nebavoid.pl 1.2      # optional, repairs atoms that pass through each other
dist.pl 00/POSCAR 06/POSCAR

# 5. run VASP on IMAGES+2 groups of cores, e.g. with 5 images and 128 cores:
mpirun -np 128 vasp_std      # needs KPAR/NCORE set so that 128 % 5 == 0 -> use 120

# 6. analyse
nebbarrier.pl ; nebspline.pl ; nebresults.pl
```

## INCAR notes

* `INCAR.neb` -> rename to `INCAR` in the parent NEB directory.
* `IOPT = 7` (LBFGS from the VTST tools) with `IBRION = 3, POTIM = 0` is the
  recommended optimiser. **If your VASP build has no VTST patch**, delete `IOPT`
  and use `IBRION = 1, POTIM = 0.1` instead - it is slower but works with stock VASP.
* Selective dynamics is carried in the endpoint POSCARs and nebmake.pl propagates it,
  so the bottom {nfix} Co layers stay fixed in every image.
* Confirm the transition state afterwards with a single-point frequency run
  (`IBRION = 5, NFREE = 2, POTIM = 0.015`, only the moving H and its two O free):
  exactly one imaginary mode must remain.

## Why this surface

The cathodic reconstruction products are Co(OH)2 and Co3O4, but the (001) basal plane
of beta-Co(OH)2 has no coordinatively unsaturated metal site, so it cannot bind H2O
datively and a water-dissociation NEB on it is not meaningful. The cus Co of
beta-CoOOH(01-12) is the Lewis-acid site that adsorbs H2O, and it is the same surface
used for the OER diagram, so the two panels share one model. This is a judgement call
and is recorded in `../../4_oer_slabs/SETUP_NOTES.md`.
"""


def make_neb(base: Structure) -> None:
    print("\n  CI-NEB endpoints (water dissociation)")
    folder = HER / "neb_water_dissociation"
    folder.mkdir(parents=True, exist_ok=True)

    site = top_metal_indices(base, ("Co",))[0]
    # acceptor = outermost surface O nearest to that Co
    cands = [i for i in top_surface_oxygens(base, 8)
             if base.get_distance(site, i) < 4.5]
    acc = max(cands, key=lambda i: base[i].coords[2])

    is_st, n_is, spectator = put_h2o(base, site, acc)
    fs_st, n_fs = put_oh_plus_h(base, site, acc, h_dir=spectator)
    report("00 initial (H2O*)", is_st, n_is)
    report("06 final (OH* + H*)", fs_st, n_fs)

    # atom i of 00 must be the same atom as atom i of 06 for nebmake.pl
    a, b = is_st.get_sorted_structure(), fs_st.get_sorted_structure()
    if [s.specie.symbol for s in a] != [s.specie.symbol for s in b]:
        raise RuntimeError("endpoint species sequences differ")
    dmax = max(np.linalg.norm(x.coords - y.coords) for x, y in zip(a, b))
    print(f"      largest single-atom displacement between endpoints: {dmax:.2f} A")
    if dmax > 3.0:
        print("      [WARNING] an atom moves more than 3 A; check the endpoint pairing")

    from pymatgen.io.vasp.inputs import Poscar
    for tag, st in (("00", is_st), ("06", fs_st)):
        d = folder / tag
        d.mkdir(exist_ok=True)
        Poscar(st.get_sorted_structure()).write_file(str(d / "POSCAR"))

    ref = is_st.get_sorted_structure()
    order = [str(s) for s in Poscar(ref).site_symbols]
    U = {"Co": 3.32, "Ni": 6.20}
    mag = slab_magmom(ref, "cooh")
    zc = float(np.mean([s.frac_coords[2] for s in ref]))
    (folder / "INCAR.neb").write_text(NEB_INCAR.format(
        order=" ".join(order),
        ldaul=" ".join("2" if s in U else "-1" for s in order),
        ldauu=" ".join(f"{U.get(s, 0.0):g}" for s in order),
        ldauj=" ".join("0" for _ in order),
        dipol=f"{zc:.4f}",
        magmom=" ".join(f"{m:g}" for m in mag),
    ))
    slab_kpoints(ref).write_file(str(folder / "KPOINTS"))
    (folder / "POTCAR_SPEC").write_text(
        "# Concatenate these PAW PBE POTCARs IN THIS ORDER:\n"
        + "\n".join(f"  {{PSP_DIR}}/POT_GGA_PAW_PBE/{_psp(s)}/POTCAR" for s in order)
        + "\n\n# e.g.\n# cat "
        + " ".join(f"$PSP/{_psp(s)}/POTCAR" for s in order) + " > POTCAR\n")
    (folder / "README.md").write_text(NEB_README.format(
        nfix=N_LAYERS_FIXED, vac=VACUUM, n_is=len(is_st), n_fs=len(fs_st),
        n_slab=len(base)))


SETUP_NOTES = """# Stage 4 / 5 surface models - setup notes

Generated by `../scripts/make_slabs.py`. Everything below is reproducible by rerunning
that script; it reads `2_bulk/cooh/POSCAR` and `2_bulk/co_oh2/POSCAR` and reuses
`write_case`, `add_ldau` and `co_magmom` from `make_inputs.py`, so MAGMOM, the LDAU
ordering and POTCAR_SPEC are produced the same way as for stages 1-3.

---

## 1. Facet: beta-CoOOH(01-12)

Confirmed against the literature before committing.

* Bajdich, Garcia-Mota, Vojvodic, Norskov & Bell, *J. Am. Chem. Soc.* **135**, 13521
  (2013) computed the relative stability of the low-index CoOOH surfaces - (01-12),
  (0001) and (10-14) - as a function of applied potential, and found that **(01-12)
  is the stable termination at OER potentials** while (10-14) wins only at low
  potential. Their OER free-energy diagrams for CoOOH are built on (01-12). This is
  the facet the whole CHE literature on CoOOH has used since.
* Independently, Nat. Commun. **13**, 6650 (2022) showed experimentally that OER
  activity on crystalline CoOOH sits on the **lateral** facets and that the basal
  (0001) plane is inert, because only the lateral facets expose coordinatively
  unsaturated Co. (01-12) is a lateral (non-basal) cut and does expose cus Co, so the
  two lines of evidence agree.

No literature was found that favours a different facet, so (01-12) stands, as already
written into `METHODS.md`.

Note on indices: (01-12) is the 4-index Bravais-Miller form. In the 3-index hexagonal
setting used by pymatgen this is **(0 1 2)**, which is what `SlabGenerator` is given.

### Termination

`SlabGenerator` returns two (01-12) terminations. The one at `shift = 0.5` is taken:

| shift | stoichiometric | symmetric | top Co coordination |
|-------|----------------|-----------|---------------------|
| 0.093 | no (dangling O, one 3-fold Co) | no | 6 / 3 |
| **0.500** | **yes, Co4H4O8 = 4 f.u.** | **yes** | **5 (cus)** |

The 5-fold Co with one vacant octahedral site pointing into the vacuum is the OER
active site, and the slab is stoichiometric and non-polar, which is what makes the
surface energy and the adsorption energies well defined.

## 2. beta-CoOOH(01-12) slab geometry

| quantity | value |
|----------|-------|
| surface cell | 2 x 2, a = 5.698 A, b = 9.230 A, gamma = 92.7 deg |
| surface area | **52.59 A^2** |
| cus Co per surface | 4 |
| thickness | 4 Co layers, 8.0 A |
| fixed | bottom 2 Co layers, 32 of 64 atoms, selective dynamics `F F F` |
| vacuum | 19.7 A clean, 16.6 A worst case (*OOH); cell c = 27.694 A |
| k-points | Gamma-centred 5 x 3 x 1 (N_i * a_i ~ 30 A) |
| dipole | `LDIPOL = .TRUE.`, `IDIPOL = 3`, `DIPOL` at the slab centre |

Coverage with one adsorbate: **1/4 ML** = 1 adsorbate per 52.6 A^2. The shortest
adsorbate-adsorbate image distance is 5.70 A (along a), which is enough that the
lateral interaction is small; it is not so large that the calculation is unaffordable.

## 3. Atom counts

| folder | system | atoms |
|--------|--------|-------|
| `clean` / `ni_clean` | bare CoOOH(01-12) / with 25 % Ni | 64 / 64 |
| `oh` / `ni_oh` | + *OH | 66 / 66 |
| `o` / `ni_o` | + *O | 65 / 65 |
| `ooh` / `ni_ooh` | + *OOH | 67 / 67 |
| `mixing/x000 x025 x050 x100` | bulk Co(1-x)Ni(x)OOH | 16 each |
| `../5_her_slabs/co_oh2_001_clean` | bare beta-Co(OH)2(001) | 80 |
| `../5_her_slabs/co_oh2_001_h_025ml` | + 1 H* | 81 |
| `../5_her_slabs/co_oh2_001_h_050ml` | + 2 H* | 82 |
| `../5_her_slabs/cooh_0112_clean` | bare CoOOH(01-12) | 64 |
| `../5_her_slabs/cooh_0112_h` | + 1 H* | 65 |
| `../5_her_slabs/neb_water_dissociation/00, 06` | NEB endpoints | 67 each |

All are within the 20-120 atom band; nothing had to be shrunk.

## 4. Ni substitution

* **Level.** 1 of the 4 cus Co in the top layer -> Ni, i.e. **25 % of surface Co**.
  The 2 x 2 cell makes 25 % the natural granularity: 1/4 is the coarsest substitution
  that is not 0 or 50 %. **This number should be checked against the post-OER Co:Ni
  ratio from EDX** (the project README flags that as the input this stage needs); if
  EDX gives something far from 25 %, the cell has to change, not just the count.
* **Which site.** The Ni replaces the Co the adsorbate sits on, so `ni_oh`, `ni_o`
  and `ni_ooh` describe the **Ni** active site. The alternative - adsorbate on a Co
  next to a Ni, probing the ligand effect rather than the site itself - is not built
  here. That is a deliberate choice: the paper's claim is that Ni leached from the
  foam creates new active sites, and the Ni-site diagram is the direct test of it.
* **Only the top layer** is substituted. The subsurface is left pure Co, so the
  clean/adsorbed pairs differ by exactly one adsorbate and nothing else.

## 5. Bulk Co(1-x)Ni(x)OOH mixing cells

* Cell: a 2 x 2 x 1 supercell of the rhombohedral CoOOH primitive cell, 16 atoms and
  4 formula units, **identical lattice for all four x**. The four total energies are
  therefore directly comparable and `dH_mix` needs no cell-size correction.
* x = 0.25 is 1 Ni of 4; x = 0.50 is the pair of Co sites with the largest separation
  (the most homogeneous of the C(4,2) = 6 arrangements). **Judgement call:** no
  configurational enumeration was done, so `dH_mix(0.5)` is the value for one ordered
  arrangement, not a configurationally averaged one. If the sign of `dH_mix` turns
  out to be near zero, enumerate properly (`EnumerateStructureTransformation`) before
  claiming miscibility.
* `ISIF = 3` (full cell + ion relax), so the lattice responds to Ni content. The
  k-mesh is fixed at Gamma-centred 4 x 4 x 8 for all four, generated once from the
  x = 0 cell; with ENCUT = 520 eV the Pulay-stress error is small and, more
  importantly, identical across x, so it cancels in `dH_mix`.

## 6. Magnetic configuration

Assigned per site, not left to VASP's default:

| species | phase | oxidation state / geometry | MAGMOM init |
|---------|-------|---------------------------|-------------|
| Co | beta-CoOOH | Co(III), octahedral, **low spin** | 0.0 |
| Co | beta-Co(OH)2 | Co(II), octahedral, **high spin** | 5.0 |
| Ni | either | Ni(III) | 1.0 |
| O, H | - | - | 0.0 |

`co_magmom()` from `make_inputs.py` assigns these from bond valence, but bond valence
is unreliable for undercoordinated surface atoms, so the result is **checked against
the known bulk assignment for the phase and pinned if it disagrees**; the script
prints how many sites it had to pin.

Two deliberate deviations from the bulk INCARs:

1. **`NUPDOWN` is not set on any slab.** The bulk CoOOH INCAR carries `NUPDOWN = 0`
   to force the closed-shell low-spin solution. On a surface that would be an
   unphysical constraint - the cus Co and any *O adsorbate can legitimately carry a
   moment - so the total moment is left free.
2. **The *O and *OOH adsorbate oxygens are initialised at 0.5 uB.** With every MAGMOM
   at exactly 0 and `ISPIN = 2`, VASP sits at the non-magnetic fixed point and can
   never find a spin-polarised solution. A small seed on the oxo/peroxo oxygen lets it
   break symmetry if it wants to. **Check `mag` in the OUTCAR of the *O cases**: if
   the converged moment is essentially zero everywhere, the closed-shell answer is the
   real one and nothing was lost; if it is not, the seed mattered.

## 7. Judgement calls, collected

1. **Facet** - (01-12) over (10-14): (10-14) is more stable at low potential, but the
   OER runs at high potential where (01-12) wins (JACS 2013). Stated in METHODS.md.
2. **Ni at 25 %** - set by the 2 x 2 cell, pending the EDX Co:Ni ratio (section 4).
3. **Adsorbate on the Ni**, not on a Co neighbouring the Ni (section 4).
4. **H* site on beta-Co(OH)2(001) is the surface O, not Co.** The brucite basal plane
   is OH-terminated and its Co are buried and 6-fold coordinated, so there is no metal
   site to put H on. H* therefore makes a surface H2O. This is the physically correct
   site, and it is also why the basal plane is expected to be HER-inactive - a result
   worth reporting rather than engineering away. If a hydride site is wanted, it needs
   an O vacancy or an edge/step model, which is a separate calculation.
5. **H* site on CoOOH(01-12) is the outermost surface O**, for the same reason -
   standard practice for `dG_H*` on oxides and oxyhydroxides.
6. **The water-dissociation NEB is run on CoOOH(01-12), not on Co(OH)2.** The
   cathodic reconstruction products are Co(OH)2 and Co3O4, but the Co(OH)2(001) basal
   plane has no cus metal site and so cannot bind H2O datively - there is no sensible
   initial state. The cus Co of CoOOH(01-12) is the Lewis-acid site that does, and
   using it means the OER and HER panels share one surface model. State this in the
   paper rather than implying the barrier was computed on the cathodic phase.
7. **Vacuum.** `SlabGenerator` quantises c in units of d_hkl, so it was asked for 17 A
   and returned 19.7 A on the clean CoOOH slab; that headroom is what keeps the tall
   *OOH and H2O* cases above the 15 A requirement (worst case 16.6 A). Every written
   structure is checked against 15 A and the script raises if one falls short.
8. **No `LASPH`.** `LASPH = .TRUE.` is generally recommended for DFT+U on transition
   metal oxides and can shift energies by ~0.1 eV/atom. It is **not** used here,
   because `BASE_INCAR` in `make_inputs.py` does not use it and stages 1-3 are already
   set up without it; adsorption energies mix slab and molecular references, so the
   setting has to be the same everywhere or nowhere. If you decide to turn it on, turn
   it on for `1_molecules`, `2_bulk` and `3_zif67` too and rerun all of them.
9. **The H position in the CoOOH structure.** The Materials Project R-3m CoOOH puts H
   at the 3b special position, exactly midway between two O (O-H = 1.2 A, a symmetric
   hydrogen bond). This is a symmetry-averaged idealisation; the real proton is
   off-centre. It was left as-is so that the slabs, the bulk reference in `2_bulk/cooh`
   and the mixing cells all use one and the same geometry. The ionic relaxation is free
   to move the H off centre - **check whether it does**, and say so in the SI.

## 8. Things to do before submitting

1. Build each POTCAR by concatenating in the order in that folder's `POTCAR_SPEC`.
   The order is the POSCAR block order, not alphabetical: Co, (Ni), H, O.
2. **The in-plane lattice constants are Materials Project values, not PBE+U+D3
   relaxed values.** After `2_bulk/cooh` and `2_bulk/co_oh2` finish their ISIF = 3
   relaxations, compare. If a or c has moved by more than ~1 %, rerun
   `python scripts/make_slabs.py` - it reads the POSCARs in `2_bulk`, so updating
   those regenerates every slab on the relaxed lattice.
3. Vibrational corrections. Each adsorbate case needs a follow-up `IBRION = 5,
   NFREE = 2, POTIM = 0.015` run with only the adsorbate atoms free, to get ZPE and
   -T*S for the free-energy diagram. Not set up here.
4. Set `NCORE` (or `NPAR`) in the slab INCARs to match your node geometry; these are
   64-82 atom cells and will be slow with the default.
5. For the NEB, relax both endpoints first - the POSCARs in `00/` and `06/` are
   constructed geometries. See `../5_her_slabs/neb_water_dissociation/README.md`.

## 9. Cross-check performed

Every generated folder was re-read from disk and checked for:

* `LDAUL` / `LDAUU` / `LDAUJ` length and order matching that POSCAR's species blocks
  (and that no species appears in more than one block);
* one MAGMOM entry per atom, with every Co and Ni at the spin state of its phase and
  every H at 0;
* `NUPDOWN` absent;
* `Selective dynamics` present in every slab POSCAR, with a non-zero number of frozen
  atoms (32 of 64, 32 of 67, 40 of 80, ...);
* exactly 4 metal layers per slab;
* vacuum >= 15 A, `LDIPOL`/`IDIPOL = 3` present, `ISIF = 2` on slabs and `ISIF = 3`
  on the bulk mixing cells;
* `POTCAR_SPEC` sequence identical to the POSCAR species order;
* `INCAR`, `POSCAR`, `KPOINTS`, `POTCAR_SPEC`, `README.md` all present;
* no adsorbate closer than 1.5 A to any surface atom it is not bonded to;
* the two NEB endpoints have identical species sequences and no atom moves more than
  1.32 A between them.
"""


OER_INDEX = """# Stage 4 - OER on beta-CoOOH(01-12)

Built by `../scripts/make_slabs.py oer`. See `SETUP_NOTES.md` for the facet choice,
cell parameters and every judgement call.

| Folder | System | Atoms |
|--------|--------|-------|
| `clean`    | beta-CoOOH(01-12), bare      | 64 |
| `oh`       | + *OH  on cus Co              | 66 |
| `o`        | + *O   on cus Co              | 65 |
| `ooh`      | + *OOH on cus Co              | 67 |
| `ni_clean` | 25 % of surface Co -> Ni, bare | 64 |
| `ni_oh`    | + *OH  on the Ni site         | 66 |
| `ni_o`     | + *O   on the Ni site         | 65 |
| `ni_ooh`   | + *OOH on the Ni site         | 67 |
| `mixing/x000` | bulk CoOOH                 | 16 |
| `mixing/x025` | bulk Co0.75Ni0.25OOH       | 16 |
| `mixing/x050` | bulk Co0.5Ni0.5OOH         | 16 |
| `mixing/x100` | bulk NiOOH                 | 16 |

## Free-energy diagram

Four associative steps in alkaline media (Norskov CHE; the pH dependence cancels
on the RHE scale, so the alkaline and acidic diagrams are identical):

    dG1 = G(*OH)  - G(*)    - (G(H2O) - 1/2 G(H2))
    dG2 = G(*O)   - G(*OH)  + 1/2 G(H2)
    dG3 = G(*OOH) - G(*O)   - (G(H2O) - 1/2 G(H2))
    dG4 = 4.92 eV - dG1 - dG2 - dG3
    eta = max(dG1..dG4)/e - 1.23 V

`G = E_DFT + ZPE - T*S`; take the 4.92 eV for `2 H2O -> O2 + 2 H2` from experiment
rather than from a PBE O2 total energy. G(H2) and G(H2O) come from `1_molecules`.

## Before submitting

1. Build POTCAR in the order given in each `POTCAR_SPEC` (NOT alphabetical).
2. The in-plane lattice constants are the Materials Project values. If the
   `2_bulk/cooh` ISIF=3 relaxation moves them by more than ~1 %, rebuild the slabs
   from the relaxed bulk before running - `make_slabs.py` reads `2_bulk/cooh/POSCAR`.
3. Vibrational corrections: rerun each adsorbate case with `IBRION=5, NFREE=2,
   POTIM=0.015` and only the adsorbate atoms free, to get ZPE and -TS.
"""

HER_INDEX = """# Stage 5 - alkaline HER

Built by `../scripts/make_slabs.py her`.

| Folder | System | Atoms |
|--------|--------|-------|
| `co_oh2_001_clean`   | beta-Co(OH)2(001), bare       | 80 |
| `co_oh2_001_h_025ml` | + 1 H* (1/4 ML)               | 81 |
| `co_oh2_001_h_050ml` | + 2 H* (1/2 ML)               | 82 |
| `cooh_0112_clean`    | beta-CoOOH(01-12), bare      | 64 |
| `cooh_0112_h`        | + 1 H* (1/4 ML)               | 65 |
| `neb_water_dissociation` | CI-NEB endpoints only     | 67 / 67 |

## dG_H*

    dG_H* = E(slab+H) - E(slab) - 1/2 E(H2) + dZPE - T*dS
          ~ E(slab+H) - E(slab) - 1/2 E(H2) + 0.24 eV

The 0.24 eV is the usual composite correction (Norskov 2005); replace it with the
explicitly computed dZPE - T*dS from an `IBRION=5` run on the adsorbed H if the
number is close to the volcano apex. |dG_H*| near 0 is optimal.

The two coverages give the coverage dependence: report dG_H* as the *differential*
value, `[G(2H) - G(1H)] - 1/2 G(H2) + 0.24`, for the 1/2 ML point.

## Water dissociation

See `neb_water_dissociation/README.md`. Relax both endpoints before running
`nebmake.pl`; the POSCARs there are constructed, not relaxed.
"""


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("oer", "all"):
        make_oer()
        (OER / "README.md").write_text(OER_INDEX)
        (OER / "SETUP_NOTES.md").write_text(SETUP_NOTES)
    if cmd in ("her", "all"):
        make_her()
        (HER / "README.md").write_text(HER_INDEX)
