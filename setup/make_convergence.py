#!/usr/bin/env python
"""Build the ENCUT / k-mesh convergence set for a stage-2 bulk phase.

    python scripts/make_convergence.py <reference_dir> <out_dir>

<reference_dir> must hold the INCAR, POSCAR, POTCAR and POTCAR_SPEC of the
converged production run whose settings are being tested.  The POSCAR is used at
fixed geometry - this is a basis-set test, not a relaxation.

Two things here are not obvious and were both learned the hard way:

ENAUG must be pinned.
    With PREC = Accurate and no explicit ENAUG, VASP derives the augmentation
    ("fine") FFT grid NGXF from ENCUT.  Sweeping ENCUT then moves NGXF as well
    (measured on beta-Co(OH)2: 48 -> 56 and 60 -> 64 -> 72 -> 80), so the
    augmentation charge is represented on a different grid at every point and
    the total energy wanders non-monotonically by ~10-20 meV instead of
    converging.  Pinning ENAUG above the largest POTCAR EAUG (605.4 eV for O)
    holds NGXF fixed and isolates the plane-wave basis, which is the whole point
    of the test.

LREAL is switched off.
    Real-space projection carries its own cutoff-dependent error of order meV,
    which sits on top of the basis-set error being measured.  For cells this
    small reciprocal-space projection is both the accurate choice and free.  One
    extra job reproduces the production LREAL = Auto setting so the cost of that
    approximation can be quoted rather than assumed.

ISYM is inherited from the reference INCAR and NOT overridden.  For beta-Co(OH)2
that matters: the symmetric hexagonal cell traps a metallic minority-spin state
0.8 eV above the orbital-ordered insulating ground state, so the convergence test
has to be run in whichever branch production actually uses.  See
2_bulk/co_oh2/PROVENANCE_symmetry.txt.
"""

import os
import re
import shutil
import sys

ENCUTS = (400, 450, 500, 520, 550, 600, 650, 700)
KSPACINGS = (0.50, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15)
ENAUG = 800  # above the largest EAUG in the POTCAR set (O, 605.392 eV)

PROD_ENCUT = 520
PROD_KSPACING = 0.25


def write_case(dest, incar, encut, kspacing, lreal, enaug=ENAUG):
    os.makedirs(dest, exist_ok=True)
    txt = incar
    subs = {
        "ENCUT": f"ENCUT = {encut}",
        "KSPACING": f"KSPACING = {kspacing}",
        "LREAL": f"LREAL = {lreal}",
        "NSW": "NSW = 0",
        "IBRION": "IBRION = -1",
        "ISIF": "ISIF = 2",
    }
    for key, line in subs.items():
        pattern = rf"^{key}\s*=.*$"
        if re.search(pattern, txt, flags=re.M):
            txt = re.sub(pattern, line, txt, flags=re.M)
        else:
            txt = txt.rstrip() + "\n" + line + "\n"
    if re.search(r"^ENAUG\s*=.*$", txt, flags=re.M):
        txt = re.sub(r"^ENAUG\s*=.*$", f"ENAUG = {enaug}", txt, flags=re.M)
    else:
        txt = txt.rstrip() + f"\nENAUG = {enaug}\n"
    with open(os.path.join(dest, "INCAR"), "w") as fh:
        fh.write(txt)
    return dest


def main(ref, out):
    incar = open(os.path.join(ref, "INCAR")).read()
    made = []

    for e in ENCUTS:
        made.append(write_case(f"{out}/encut_{e}", incar, e, PROD_KSPACING, ".FALSE."))
    for k in KSPACINGS:
        tag = str(k).replace(".", "p")
        made.append(write_case(f"{out}/kspacing_{tag}", incar, PROD_ENCUT, k, ".FALSE."))
    made.append(
        write_case(f"{out}/prod_lreal_auto", incar, PROD_ENCUT, PROD_KSPACING, "Auto")
    )

    for d in made:
        for f in ("POSCAR", "POTCAR", "POTCAR_SPEC"):
            shutil.copy(os.path.join(ref, f), os.path.join(d, f))

    with open(f"{out}/JOBLIST", "w") as fh:
        fh.write("\n".join(made) + "\n")
    print(f"{len(made)} jobs written under {out}/  (reference: {ref})")
    print(f"submit with:\n  ZIF_NTASKS=16 ZIF_TIME=0:30:00 "
          f"bash submit/submit_these.sh $(cat {out}/JOBLIST)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
