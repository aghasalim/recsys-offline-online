"""House plot style. Import before plotting, it works by side effect.

One place to change how every figure in the repo looks, so the plots in the
README read as one set rather than eight unrelated ones.
"""
from __future__ import annotations

import matplotlib as mpl

# Picked for contrast in greyscale and for deuteranopia, since a reader may
# print this or may not see red and green apart.
PALETTE = ["#1f4e79", "#b2182b", "#1a9850", "#d9822b", "#6a51a3", "#4d4d4d"]

mpl.rcParams.update({
    "figure.dpi": 170,
    "savefig.dpi": 170,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.titlesize": 12.5,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "axes.labelsize": 10.5,
    "axes.labelcolor": "#222222",
    "axes.edgecolor": "#bbbbbb",
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#dedede",
    "grid.linewidth": 0.7,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.frameon": False,
    "legend.fontsize": 9.5,
    "lines.linewidth": 2.0,
    "lines.markersize": 4.5,
    "axes.prop_cycle": mpl.cycler(color=PALETTE),
})


def titled(ax, title: str, subtitle: str = "") -> None:
    """Bold title with a grey explanatory line under it."""
    ax.set_title(title, pad=26 if subtitle else 10)
    if subtitle:
        ax.text(0.0, 1.012, subtitle, transform=ax.transAxes, fontsize=9.3,
                color="#5a5a5a", va="bottom", ha="left")
