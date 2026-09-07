"""Figure style.

Sans-serif throughout, hairline closed-box panels, inward ticks with minor
ticks shown, and the colour-blind-safe palette of Okabe and Ito. Column widths are
88 mm single and 180 mm double. Text is kept at or below 7 pt and vector output
embeds fonts rather than outlining them.
"""
from __future__ import annotations
__author__ = "Nabil Khossossi"

from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm

# --------------------------------------------------------------------------- #
#  Colour system
# --------------------------------------------------------------------------- #
PALETTE = {
    "ink":       "#1A1A1A",
    "subtle":    "#6B6B6B",
    "hair":      "#D8D8D8",
    "blue":      "#0B66B2",
    "teal":      "#1B9E89",
    "green":     "#2E8B57",
    "amber":     "#E69F00",
    "vermilion": "#D55E00",
    "purple":    "#7B4FA3",
    "rose":      "#CC5C7A",
    "navy":      "#173B5E",
}

# Fixed identity colours, assigned once and never cycled.
C_PRIMARY   = PALETTE["blue"]        # pristine / calculated / Co
C_SECONDARY = PALETTE["vermilion"]   # corrected / measured / O
C_TERTIARY  = PALETTE["teal"]        # Ni-containing
C_QUAT      = PALETTE["purple"]      # Ni 3d specifically
INK, SUBTLE, HAIR = PALETTE["ink"], PALETTE["subtle"], PALETTE["hair"]

# Column widths in inches.
COL_SINGLE  = 88 / 25.4    # 88 mm single column
COL_ONEHALF = 120 / 25.4
COL_DOUBLE  = 180 / 25.4   # 180 mm, double column


def _pick_serif() -> list[str]:
    available = {f.name for f in fm.fontManager.ttflist}
    preferred = ["Charter", "Bitstream Charter", "Georgia", "Source Serif Pro",
                 "STIXGeneral", "DejaVu Serif"]
    ordered = [f for f in preferred if f in available]
    return ordered or ["DejaVu Serif"]


def _pick_sans() -> list[str]:
    available = {f.name for f in fm.fontManager.ttflist}
    # Helvetica on macOS often registers only the regular face, so a bold
    # request silently falls back to regular. Prefer families that ship a
    # real bold so panel labels stay bold.
    preferred = ["Arial", "Helvetica", "Source Sans Pro", "Open Sans",
                 "TeX Gyre Heros", "DejaVu Sans"]
    ordered = [f for f in preferred if f in available]
    return ordered or ["DejaVu Sans"]


def apply_style(base_fontsize: float = 7.0) -> None:
    """Install the design system globally. Call once per script."""
    serif, sans = _pick_serif(), _pick_sans()
    mpl.rcParams.update({
        # Sans-serif throughout, Helvetica or Arial. Max 7 pt, min 5 pt.
        "font.family": "sans-serif",
        "font.serif": serif,
        "font.sans-serif": sans,
        "mathtext.fontset": "cm",
        "font.size": base_fontsize,
        "axes.labelsize": base_fontsize,
        "axes.titlesize": base_fontsize + 1,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,
        "legend.fontsize": base_fontsize - 1,
        "figure.titlesize": base_fontsize + 2,
        "axes.linewidth": 0.7,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "axes.grid": False,
        "grid.color": HAIR,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0,
        "legend.labelspacing": 0.35,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 140,
    })


def panel_label(ax, letter: str, dx: float = -0.085, dy: float = 1.02,
                fontsize: float = 8.0, weight: str = "bold") -> None:
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=fontsize,
            fontweight=weight, va="bottom", ha="right",
            family=_pick_sans(), color=INK)


def tidy(ax) -> None:
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.7)
    ax.tick_params(which="both", top=False, right=False)


def framed(ax, lw: float = 0.8) -> None:
    """Closed-box panel: four hairline spines, ticks inward on left+bottom."""
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(lw)
        ax.spines[s].set_color(INK)
    ax.tick_params(which="both", top=False, right=False, direction="in")


def no_minor(ax, axis: str = "both") -> None:
    """Categorical axes must not carry minor ticks."""
    from matplotlib.ticker import NullLocator
    if axis in ("x", "both"):
        ax.xaxis.set_minor_locator(NullLocator())
    if axis in ("y", "both"):
        ax.yaxis.set_minor_locator(NullLocator())


def save(fig, name: str, outdir: Path) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.pdf")
    fig.savefig(outdir / f"{name}.png", dpi=600)
    return outdir / f"{name}.pdf"
