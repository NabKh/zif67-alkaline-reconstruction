"""Command-line entry point.

    python -m zif67 --root /path/to/calculations oer
    python -m zif67 --root /path/to/calculations all

The calculation tree is not distributed with this repository because the VASP
output files are large; it is kept separately (see README).
"""
__author__ = "Nabil Khossossi"
import argparse

from . import bulk, electronic, her, oer

ANALYSES = {
    "bulk": bulk.report,
    "oer": oer.report,
    "her": her.report,
    "electronic": electronic.report,
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="zif67", description=__doc__)
    parser.add_argument("analysis", choices=[*ANALYSES, "all"])
    parser.add_argument("--root", required=True,
                        help="root of the calculation tree")
    args = parser.parse_args()

    names = list(ANALYSES) if args.analysis == "all" else [args.analysis]
    for i, name in enumerate(names):
        if i:
            print("\n" + "-" * 70 + "\n")
        ANALYSES[name](args.root)


if __name__ == "__main__":
    main()
