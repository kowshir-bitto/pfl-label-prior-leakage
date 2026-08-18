"""Publication figure styling and shared rendering helpers (300 DPI)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch  # noqa: F401  (re-exported for scripts)

from .config import CFG, FIG_DIR

PAL = list(CFG.fig.palette)


def use_paper_style() -> None:
    """Single entry point so every figure in the paper looks identical."""
    try:
        plt.style.use(CFG.fig.style)
    except OSError:
        plt.style.use("seaborn-v0_8-paper")
    sns.set_palette(PAL)
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": CFG.fig.dpi,
        "font.family": "sans-serif",
        "font.sans-serif": [CFG.fig.font_family, "Arial", "Helvetica"],
        "font.size": CFG.fig.base_fontsize,
        "axes.titlesize": CFG.fig.base_fontsize + 1,
        "axes.labelsize": CFG.fig.base_fontsize,
        "xtick.labelsize": CFG.fig.base_fontsize - 1,
        "ytick.labelsize": CFG.fig.base_fontsize - 1,
        "legend.fontsize": CFG.fig.base_fontsize - 1,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,     # editable text in the PDF (journal requirement)
        "ps.fonttype": 42,
    })


def save_fig(fig, name: str, tight: bool = True) -> list[Path]:
    """Write both a 300 DPI PNG and a vector PDF; return the written paths."""
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass
    out = []
    if CFG.fig.save_png:
        p = FIG_DIR / f"{name}.png"
        fig.savefig(p, dpi=CFG.fig.dpi, bbox_inches="tight", facecolor="white")
        out.append(p)
    if CFG.fig.save_pdf:
        p = FIG_DIR / f"{name}.pdf"
        fig.savefig(p, bbox_inches="tight", facecolor="white")
        out.append(p)
    plt.close(fig)
    return out


def method_color(method: str) -> str:
    """Proposed method always gets the accent colour; baselines cycle."""
    from .config import METHODS, PROPOSED_METHOD
    if method == PROPOSED_METHOD or method.startswith("FedGIM"):
        return CFG.fig.proposed_color
    order = [m for m in METHODS if m != PROPOSED_METHOD]
    return PAL[order.index(method) % len(PAL)] if method in order else PAL[-1]


def sig_stars(p: float) -> str:
    """APA-style significance markers used across all comparison figures."""
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def annotate_bars(ax, xs, heights, errs, labels, dy: float = 0.01,
                  fontsize: int = 6) -> None:
    for x, h, e, lab in zip(xs, heights, errs, labels):
        if lab in ("", None):
            continue
        ax.text(x, h + (e or 0) + dy, lab, ha="center", va="bottom",
                fontsize=fontsize, fontweight="bold")


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return img[:, :, ::-1]


def panel_label(ax, letter: str, dx: float = -0.08, dy: float = 1.06) -> None:
    """(a), (b), (c) ... panel tags in the journal house style."""
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes,
            fontsize=CFG.fig.base_fontsize + 1, fontweight="bold",
            va="top", ha="right")
