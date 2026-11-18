"""Wykresy funkcji przynależności IT2: UMF, LMF i FOU."""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from core.variables import LinguisticVariable
from visualization.style import PALETTE


def plot_variable(var: LinguisticVariable, x_current: Optional[float] = None, title: Optional[str] = None,
                  xlabel: Optional[str] = None, figsize=(9, 3.6)):
    """UMF (linia ciągła), LMF (przerywana), FOU (wypełnienie); opcjonalnie bieżące wejście."""
    fig, ax = plt.subplots(figsize=figsize)
    x = var.grid(1201)
    lo, hi = var.memberships(x)
    for i, term in enumerate(var.terms):
        c = PALETTE[i % len(PALETTE)]
        ax.fill_between(x, lo[i], hi[i], color=c, alpha=0.22, linewidth=0)
        ax.plot(x, hi[i], color=c, lw=1.8)
        ax.plot(x, lo[i], color=c, lw=1.2, ls="--")
        lo_u, hi_u = var.universe
        if not np.isfinite(term.a):
            pos = 0.5 * (lo_u + term.c)
        elif not np.isfinite(term.d):
            pos = 0.5 * (term.b + hi_u)
        else:
            pos = 0.5 * (term.b + term.c)
        ax.text(pos, 1.04, term.name, color=c, ha="center", fontsize=8.5)
    if x_current is not None:
        xc = float(var.clip(x_current))
        ax.axvline(xc, color="#222", lw=1.3, ls=":")
        mlo, mhi = var.memberships(xc)
        for i, (l, h) in enumerate(zip(mlo, mhi)):
            if h > 0:
                ax.plot([xc, xc], [l, h], color=PALETTE[i % len(PALETTE)], lw=4, solid_capstyle="round")
    ax.set_xlim(*var.universe)
    ax.set_ylim(-0.02, 1.15)
    default_label = f"{var.name} [{var.unit}]" if var.unit else var.name
    ax.set_xlabel(xlabel or default_label)
    ax.set_ylabel("Membership degree")
    ax.set_title(title or var.name)
    handles = [Line2D([0], [0], color="#444", lw=1.8, label="UMF"),
               Line2D([0], [0], color="#444", lw=1.2, ls="--", label="LMF"),
               Patch(facecolor="#444", alpha=0.25, label="FOU")]
    if x_current is not None:
        handles.append(Line2D([0], [0], color="#222", lw=1.3, ls=":",
                                  label=f"current input = {float(var.clip(x_current)):.1f}"))
    ax.legend(handles=handles, loc="upper right", ncol=4, bbox_to_anchor=(1, 1.22))
    fig.tight_layout()
    return fig


def plot_temperature(var, e_current=None):
    return plot_variable(var, e_current, "Temperature error membership functions (IT2)",
                         "Temperature error e = T_room − T_set [°C]")


def plot_humidity(var, h_current=None):
    return plot_variable(var, h_current, "Humidity membership functions (IT2)", "Relative humidity [%]")


def plot_output(var, y_current=None):
    return plot_variable(var, y_current, "Output membership functions (IT2)",
                         "Control output u  (−100 = max heating … +100 = max cooling)")
