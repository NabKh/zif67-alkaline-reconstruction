"""Build an ordered, DFT-ready ZIF-67 cell from the experimental refinement.

Two things make this more than a straight CIF read.

1. The refinement carries methyl-hydrogen rotational disorder: each CH3 is refined
   over two 60-degree rotamers at half occupancy, giving 348 sites instead of 276.
   A periodic DFT calculation needs an ordered cell, so one rotamer is selected per
   methyl group.
2. The order of operations matters. Choosing rotamers independently breaks the
   I-centring, so the cell is reduced to primitive first and the disorder resolved
   there; done the other way round the cell stays at 276 sites.

Source CIF: COD 4124528, ZIF-67 = Co(mim)2, I-43m, a = 16.9077 A
[Kwon, Jeong, Lee, An & Lee, J. Am. Chem. Soc. 137, 12304 (2015)].

Writes:
    structures/zif67_conventional.cif   276 atoms  (Z = 12)
    structures/zif67_primitive.cif      138 atoms  (Z = 6)  <- used for DFT
"""

__author__ = "Nabil Khossossi"

import warnings
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator

HERE = Path(__file__).parent
OUT = HERE.parent / "structures"
CIF_IN = HERE.parent / "structures" / "COD_4124528_ZIF-67.cif"

# Experimental ZIF-67 lattice constant (Banerjee et al.). The manuscript's six
# reflections index to this value to within 0.1 deg -- see docs/01_manuscript_review.md
A_ZIF67 = 16.9077

# Manuscript's reported reflections, for the validation check
REPORTED = [7.3, 10.4, 12.7, 14.8, 16.4, 18.0]


def order_methyl_hydrogens(struct: Structure) -> Structure:
    """Resolve CH3 rotational disorder by keeping one rotamer per methyl group.

    Each methyl carbon carries six half-occupied H sites: two staggered rotamers
    of three. The discriminator is geometric -- within a genuine rotamer all three
    H-H distances are EQUAL (here 1.60 A, from C-H = 0.98 A at ~109.5 deg), while
    any cross-rotamer triple has unequal distances (mixtures of 0.89 / 0.95 / 1.85).

    So: enumerate all 20 three-subsets and keep the one whose pairwise distances
    have the smallest spread. A distance threshold alone is NOT sufficient -- the
    1.85 A cross-rotamer pair passes any cutoff that admits the real 1.60 A one.
    """
    from itertools import combinations

    partial = [i for i, s in enumerate(struct)
               if abs(sum(s.species.values()) - 1.0) > 1e-3]
    full = [i for i in range(len(struct)) if i not in partial]

    carbons = [i for i in full
               if list(struct[i].species.keys())[0].symbol == "C"]
    groups: dict[int, list[int]] = {}
    for h in partial:
        _, nearest_c = min((struct.get_distance(h, c), c) for c in carbons)
        groups.setdefault(nearest_c, []).append(h)

    keep: list[int] = []
    spreads = []
    for c, hs in groups.items():
        if len(hs) != 6:
            raise RuntimeError(f"C{c}: {len(hs)} partial H, expected 6")
        best, best_spread = None, np.inf
        for trio in combinations(hs, 3):
            d = [struct.get_distance(a, b) for a, b in combinations(trio, 2)]
            spread = float(np.std(d))
            if spread < best_spread:
                best, best_spread = trio, spread
        keep.extend(best)
        spreads.append(best_spread)
    print(f"  {len(groups)} methyl groups, one rotamer each; "
          f"max H-H spread in a kept trio = {max(spreads):.4f} A")

    drop = sorted(set(partial) - set(keep), reverse=True)
    out = struct.copy()
    out.remove_sites(drop)
    for i, site in enumerate(out):
        if abs(sum(site.species.values()) - 1.0) > 1e-3:
            out.replace(i, list(site.species.keys())[0])
    return out


def report_bonds(struct: Structure, metal: str) -> None:
    def shortest(a: str, b: str) -> float:
        ds = []
        for i, si in enumerate(struct):
            if list(si.species.keys())[0].symbol != a:
                continue
            for j, sj in enumerate(struct):
                if i != j and list(sj.species.keys())[0].symbol == b:
                    ds.append(struct.get_distance(i, j))
        return float(np.min(ds)) if ds else float("nan")

    print(f"    {metal}-N  {shortest(metal, 'N'):.3f} A   (expect ~1.98-2.00)")
    print(f"    C-N   {shortest('C', 'N'):.3f} A   (expect ~1.33-1.38)")
    print(f"    C-H   {shortest('C', 'H'):.3f} A   (expect ~0.95-1.10)")
    print(f"    H-H   {shortest('H', 'H'):.3f} A   (expect > 1.5 if ordering worked)")


def main() -> None:
    warnings.simplefilter("ignore")
    print(f"Reading {CIF_IN.name}")
    raw = Structure.from_file(CIF_IN, primitive=False)
    print(f"  as read: {len(raw)} sites, a = {raw.lattice.a:.4f} A, "
          f"ordered = {raw.is_ordered}  (348 = 276 + 72 extra disordered methyl H)")

    print("\nSetting the experimental ZIF-67 lattice constant...")
    raw.scale_lattice(A_ZIF67 ** 3)
    print(f"  a = {raw.lattice.a:.4f} A  (experimental ZIF-67)")

    # ORDER OF OPERATIONS MATTERS.
    # Choosing methyl rotamers independently breaks the I-centring, after which
    # get_primitive_structure() can no longer halve the cell (returns 276, not 138).
    # So reduce to the primitive cell FIRST -- the disordered cell is still
    # I-centred -- and resolve the disorder there.
    print("\nReducing to the primitive cell BEFORE ordering...")
    prim_raw = raw.get_primitive_structure()
    print(f"  {len(raw)} -> {len(prim_raw)} sites (I-centred, exactly half)")

    print("\nResolving methyl-H disorder in the primitive cell...")
    prim = order_methyl_hydrogens(prim_raw)
    print(f"  -> {len(prim)} atoms, {prim.composition.reduced_formula}, "
          f"ordered = {prim.is_ordered}")
    if len(prim) != 138:
        raise SystemExit(f"expected 138 atoms in the primitive cell, got {len(prim)}")

    print("\nAlso building the ordered conventional cell (for XRD / visualisation)...")
    conv = order_methyl_hydrogens(raw)
    print(f"  -> {len(conv)} atoms")

    print("\nBond-length sanity check (primitive cell):")
    report_bonds(prim, "Co")
    print("    note: C-H is short because X-ray refinement underestimates H")
    print("          positions; DFT relaxation will lengthen it to ~1.09 A.")

    out_prim = OUT / "zif67_primitive.cif"
    out_conv = OUT / "zif67_conventional.cif"
    prim.to(filename=str(out_prim))
    conv.to(filename=str(out_conv))
    print(f"\n  wrote {out_prim.name} ({len(prim)} atoms)  <- USE THIS FOR DFT")
    print(f"  wrote {out_conv.name} ({len(conv)} atoms)")

    print("\nXRD validation against the manuscript's reported reflections:")
    patt = XRDCalculator(wavelength="CuKa").get_pattern(conv, two_theta_range=(5, 20))
    print("    2theta_calc   I_rel   reported   delta")
    ok = 0
    for tt, inten in zip(patt.x, patt.y):
        if inten < 2.0:
            continue
        near = min(REPORTED, key=lambda r: abs(r - tt))
        d = tt - near
        mark = "  OK" if abs(d) < 0.25 else ""
        ok += 1 if mark else 0
        print(f"    {tt:9.2f} {inten:8.1f} {near:9.1f} {d:+8.2f}{mark}")
    print(f"\n  {ok}/{len(REPORTED)} of the manuscript's reflections reproduced.")


if __name__ == "__main__":
    main()
