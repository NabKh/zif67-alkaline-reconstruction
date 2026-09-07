"""Generate VASP inputs for the ZIF-67 reconstruction project.

Everything except ZIF-67 is a 1-15 atom system. The point of this script is to get
the fiddly parts right automatically:

  * MAGMOM per site, from the correct spin state (Co(II) HS vs Co(III) LS)
  * LDAUL / LDAUU / LDAUJ arrays ordered to match the POSCAR species order
  * primitive-cell reduction (ZIF-67: 276 -> 138 atoms)
  * Gamma-only vs KSPACING chosen per system size

Usage:
    export MP_API_KEY="..."
    python make_inputs.py molecules      # stage 1: H2, H2O, Hmim
    python make_inputs.py bulks          # stage 2: CoOOH, Co(OH)2, Co3O4, Co
    python make_inputs.py zif67 PATH.cif # stage 3: ZIF-67 (FM and AFM)
    python make_inputs.py all PATH.cif
"""

__author__ = "Nabil Khossossi"

import os
import sys
from pathlib import Path

from pymatgen.core import Molecule, Structure
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar

ROOT = Path(__file__).resolve().parents[1]      # ZIF67_calculations/

# --- MP-compatible DFT+U (see docs/05_dft_methods.md) ---
U_VALUES = {"Co": 3.32, "Ni": 6.20}

BASE_INCAR = {
    "PREC": "Accurate", "ENCUT": 520, "EDIFF": 1e-6, "EDIFFG": -0.02,
    "NELM": 200, "ALGO": "Normal", "LREAL": "Auto",
    "ISMEAR": 0, "SIGMA": 0.05, "ISPIN": 2,
    "IBRION": 2, "NSW": 200, "ISIF": 3,
    "IVDW": 12,                # D3(BJ) -- mandatory for ZIF-67, harmless elsewhere
    "LMAXMIX": 4, "LORBIT": 11,
    "LWAVE": False, "LCHARG": False,
}


def add_ldau(incar: dict, structure: Structure) -> dict:
    """Attach LDAU arrays ordered to match the POSCAR species order.

    This is the step people get wrong by hand: VASP reads LDAUL/U/J in the order
    species appear in POSCAR, not alphabetically.
    """
    species = [str(s) for s in Poscar(structure).site_symbols]
    if not any(sp in U_VALUES for sp in species):
        return incar
    # MP applies U only to oxides/nitrides/fluorides, never to elemental metals.
    if not any(sp in ("O", "N", "F", "S") for sp in species):
        return incar
    incar.update({
        "LDAU": True, "LDAUTYPE": 2, "LDAUPRINT": 1,
        "LDAUL": [2 if sp in U_VALUES else -1 for sp in species],
        "LDAUU": [U_VALUES.get(sp, 0.0) for sp in species],
        "LDAUJ": [0.0 for _ in species],
    })
    incar["_species_order"] = " ".join(species)      # stripped before writing
    return incar


def co_magmom(structure: Structure, afm: bool = False) -> list:
    """Per-site MAGMOM from oxidation state + coordination.

    Coordination number alone is NOT enough: octahedral Co in Co(OH)2 is Co(II)
    high spin (~3 uB) while octahedral Co in Co3O4 and CoOOH is Co(III) LOW spin
    (0 uB). Getting this wrong is worth ~0.5 eV, so the oxidation state is
    assigned by bond valence and only falls back to coordination.

        Co(II)  any geometry   -> high spin, init 5.0
        Co(III) octahedral     -> LOW spin,  init 0.0
        Co(III) tetrahedral    -> high spin, init 4.0  (rare)
        Ni(II) / Ni(III)       -> 2.0 / 1.0
    """
    import warnings

    from pymatgen.analysis.bond_valence import BVAnalyzer
    from pymatgen.analysis.local_env import CrystalNN

    # Elemental metal: ferromagnetic, experimental bulk moments.
    if len(structure.composition.elements) == 1:
        el = structure.composition.elements[0].symbol
        m0 = {"Co": 2.0, "Ni": 0.7}.get(el, 0.0)
        return [m0] * len(structure)

    valences = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            valences = BVAnalyzer().get_valences(structure)
        except Exception:
            pass

        cnn = CrystalNN()
        mags, flip = [], 1
        for i, site in enumerate(structure):
            el = site.specie.symbol
            if el not in ("Co", "Ni"):
                mags.append(0.0)
                continue
            try:
                cn = cnn.get_cn(structure, i)
            except Exception:
                cn = 6
            ox = valences[i] if valences else (2 if cn <= 4 else 3)

            if el == "Ni":
                m = 2.0 if ox <= 2 else 1.0
            elif ox <= 2:
                m = 5.0                       # Co(II) high spin
            elif ox >= 3 and cn >= 5:
                m = 0.0                       # Co(III) octahedral LOW spin
            else:
                m = 4.0                       # Co(III) tetrahedral high spin
            if m > 0:
                mags.append(m * flip)
                if afm:
                    flip *= -1
            else:
                mags.append(0.0)
    return mags


def write_case(name: str, structure: Structure, folder: Path, *,
               gamma_only: bool = False, kspacing: float = 0.25,
               afm: bool = False, relax_cell: bool = True,
               extra: dict | None = None) -> None:
    folder.mkdir(parents=True, exist_ok=True)

    incar = dict(BASE_INCAR)
    incar["ISIF"] = 3 if relax_cell else 2
    incar = add_ldau(incar, structure)

    mags = co_magmom(structure, afm=afm)
    incar["MAGMOM"] = mags
    if all(abs(m) < 1e-6 for m in mags):
        incar["NUPDOWN"] = int(0)          # enforce the low-spin / closed-shell solution

    if gamma_only:
        Kpoints.gamma_automatic((1, 1, 1)).write_file(str(folder / "KPOINTS"))
    else:
        # VASP KSPACING is a minimum k-point spacing in 1/Angstrom (NOT a
        # reciprocal density). 0.25 gives ~11x11x11 for CoOOH; 0.03 would give
        # 87x87x87. Converge this on co_oh2 before production runs.
        incar["KSPACING"] = kspacing
        incar["KGAMMA"] = True
    if extra:
        incar.update(extra)

    species_note = incar.pop("_species_order", None)
    Incar(incar).write_file(str(folder / "INCAR"))
    Poscar(structure).write_file(str(folder / "POSCAR"))

    order = [str(s) for s in Poscar(structure).site_symbols]
    (folder / "POTCAR_SPEC").write_text(
        "# Concatenate these PAW PBE POTCARs IN THIS ORDER:\n"
        + "\n".join(f"  {{PSP_DIR}}/POT_GGA_PAW_PBE/{_psp(s)}/POTCAR" for s in order)
        + "\n\n# e.g.\n# cat "
        + " ".join(f"$PSP/{_psp(s)}/POTCAR" for s in order) + " > POTCAR\n"
    )

    print(f"  {name:26s} {len(structure):4d} atoms  species={'+'.join(order)}"
          f"  {'Gamma' if gamma_only else f'KSPACING={kspacing}'}"
          f"  MAGMOM={_fmt_mag(mags)}")
    if species_note:
        print(f"  {'':26s}      LDAU order -> {species_note}")


def _psp(sym: str) -> str:
    return {"Co": "Co_pv", "Ni": "Ni_pv"}.get(sym, sym)


def _fmt_mag(mags) -> str:
    uniq = sorted({round(m, 1) for m in mags})
    return "/".join(f"{u:g}" for u in uniq)


# ---------------------------------------------------------------- stage 1
def make_molecules() -> None:
    """H2, H2O, Hmim in a 15 A box, Gamma point. Seconds to minutes each."""
    print("\nSTAGE 1 - molecular references (15 A box, Gamma)")
    from ase.build import molecule as ase_mol

    out = ROOT / "1_molecules"
    box = 15.0

    for name in ("H2", "H2O"):
        atoms = ase_mol(name)
        mol = Molecule([a.symbol for a in atoms], atoms.get_positions())
        struct = mol.get_boxed_structure(box, box, box)
        write_case(name, struct, out / name.lower(),
                   gamma_only=True, relax_cell=False,
                   extra={"SIGMA": 0.01, "LREAL": False})

    # 2-methylimidazole (Hmim), C4H6N2, 12 atoms -- built from SMILES
    from rdkit import Chem
    from rdkit.Chem import AllChem

    rd = Chem.AddHs(Chem.MolFromSmiles("Cc1ncc[nH]1"))
    AllChem.EmbedMolecule(rd, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(rd)
    conf = rd.GetConformer()
    mol = Molecule(
        [a.GetSymbol() for a in rd.GetAtoms()],
        [list(conf.GetAtomPosition(i)) for i in range(rd.GetNumAtoms())],
    )
    struct = mol.get_boxed_structure(box, box, box)
    write_case("Hmim (C4H6N2)", struct, out / "hmim",
               gamma_only=True, relax_cell=False,
               extra={"SIGMA": 0.01, "LREAL": False})
    print("  note: rerun hmim with LSOL=.TRUE., EB_K=78.4 on the relaxed geometry")


# ---------------------------------------------------------------- stage 2
BULKS = {
    "cooh":   ("CoOOH",   "R-3m",  ROOT / "2_bulk" / "cooh"),
    "co_oh2": ("Co(OH)2", "P-3m1", ROOT / "2_bulk" / "co_oh2"),
    "co3o4":  ("Co3O4",   None,    ROOT / "2_bulk" / "co3o4"),
    "co":     ("Co",      "P6_3/mmc", ROOT / "2_bulk" / "co_metal"),
}


def make_bulks() -> None:
    """All 1-14 atoms. Minutes each."""
    print("\nSTAGE 2 - bulk phases (primitive cells, all tiny)")
    from mp_api.client import MPRester

    key = os.environ.get("MP_API_KEY")
    if not key:
        raise SystemExit("Set MP_API_KEY first")

    # NOTE: 'material_id' is omitted from fields -- the installed emmet-core
    # rejects MP's new alphanumeric IDs. Provenance is recorded via formula+spg.
    fields = ["formula_pretty", "symmetry", "energy_above_hull", "structure"]

    with MPRester(key) as mpr:
        for label, (formula, want_spg, folder) in BULKS.items():
            docs = mpr.materials.summary.search(formula=formula, fields=fields)
            if not docs:
                print(f"  {label}: NOT FOUND")
                continue
            if want_spg:
                match = [d for d in docs if d.symmetry.symbol == want_spg]
                if match:
                    docs = match
                else:
                    print(f"  [warn] {label}: no {want_spg} polymorph; using lowest-energy")
            doc = sorted(docs, key=lambda d: d.energy_above_hull)[0]
            prim = doc.structure.get_primitive_structure()
            write_case(f"{formula} ({doc.symmetry.symbol})", prim, folder,
                       kspacing=0.25)
            (folder / "PROVENANCE.txt").write_text(
                f"Materials Project\nformula: {doc.formula_pretty}\n"
                f"spacegroup: {doc.symmetry.symbol}\n"
                f"E_above_hull: {doc.energy_above_hull:.4f} eV/atom\n"
                f"conventional: {len(doc.structure)} atoms -> primitive: {len(prim)} atoms\n"
            )


# ---------------------------------------------------------------- stage 3
def make_zif67(cif_path: str) -> None:
    """The only large system. Primitive reduction halves it: 276 -> 138 atoms."""
    print("\nSTAGE 3 - ZIF-67 (the only big system)")
    struct = Structure.from_file(cif_path)
    print(f"  as read:    {struct.composition.reduced_formula}, {len(struct)} atoms, "
          f"a = {struct.lattice.a:.3f} A, {struct.get_space_group_info()}")

    prim = struct.get_primitive_structure()
    print(f"  primitive:  {len(prim)} atoms   <- use this one")

    out = ROOT / "3_zif67"
    # Ions-only relax at the experimental lattice constant first -- much cheaper
    # than a full cell relax, and enough for the reaction energy.
    for tag, afm in (("fm", False), ("afm", True)):
        write_case(f"ZIF-67 {tag.upper()}", prim, out / tag,
                   gamma_only=True, afm=afm, relax_cell=False,
                   extra={"EDIFF": 1e-5, "EDIFFG": -0.03, "ALGO": "Fast", "NSW": 300})
    print("  run BOTH; the lower energy is the magnetic ground state (report which).")
    print("  only after that, optionally rerun the winner with ISIF=3.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    cif = sys.argv[2] if len(sys.argv) > 2 else None

    if cmd in ("molecules", "all"):
        make_molecules()
    if cmd in ("bulks", "all"):
        make_bulks()
    if cmd in ("zif67", "all"):
        if not cif:
            print("\nSTAGE 3 skipped - needs the ordered ZIF-67 CIF from build_zif67.py:")
            print("  python make_inputs.py zif67 /path/to/zif67.cif")
        else:
            make_zif67(cif)
    if cmd == "help":
        print(__doc__)
