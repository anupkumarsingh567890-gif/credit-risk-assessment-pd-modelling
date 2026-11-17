"""
Shared chart style so every figure in the project looks the same.

Colours follow a colour-blind-checked palette. Categorical colours are used
in a fixed order; risk bands use the status colours (good / warning /
critical) and always carry a text label too.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save files without opening windows
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from src.config import FIGURES_DIR  # noqa: E402

# Categorical slots, always used in this order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE = SERIES[0], SERIES[1]

# Status colours for risk bands
RISK_COLORS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#d03b3b"}

TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e5e4df"
SURFACE = "#fcfcfb"

SEQUENTIAL = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#6da7ec", "#1c5cab", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": TEXT_MUTED,
            "axes.edgecolor": GRID,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "lines.linewidth": 2,
            "legend.frameon": False,
        }
    )


def save(fig, name: str, subdir: str = "") -> Path:
    folder = FIGURES_DIR / subdir if subdir else FIGURES_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def bar_rate(table, label_col: str, title: str, overall: float | None = None,
             horizontal: bool = False, rate_col: str = "default_rate_pct"):
    """Bar chart of a default rate per group, with the portfolio average as a dashed line."""
    set_style()
    n = len(table)
    if horizontal:
        fig, ax = plt.subplots(figsize=(7, 0.35 * n + 1.2))
        ax.barh(table[label_col].astype(str), table[rate_col], color=BLUE, height=0.7)
        ax.invert_yaxis()
        ax.set_xlabel("Default rate (%)")
        ax.grid(axis="y", visible=False)
        for y, v in enumerate(table[rate_col]):
            ax.text(v + 0.3, y, f"{v:.1f}%", va="center", fontsize=8, color=TEXT_MUTED)
        if overall is not None:
            ax.axvline(overall, color=TEXT_MUTED, ls="--", lw=1)
    else:
        fig, ax = plt.subplots(figsize=(max(5, 0.55 * n + 2), 3.8))
        ax.bar(table[label_col].astype(str), table[rate_col], color=BLUE, width=0.7)
        ax.set_ylabel("Default rate (%)")
        ax.grid(axis="x", visible=False)
        if n <= 12:
            for x, v in enumerate(table[rate_col]):
                ax.text(x, v + 0.3, f"{v:.1f}%", ha="center", fontsize=8, color=TEXT_MUTED)
        if n > 6:
            plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        if overall is not None:
            ax.axhline(overall, color=TEXT_MUTED, ls="--", lw=1)
    if overall is not None:
        ax.set_title(f"{title}\n(dashed line = average {overall:.1f}%)", fontsize=11)
    else:
        ax.set_title(title)
    return fig, ax
