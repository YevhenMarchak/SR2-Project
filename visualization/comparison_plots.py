"""Wykresy porównawcze IT2FLS vs ON/OFF oraz wpływ niepewności (FOU)."""
from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.style import FUZZY, NEUTRAL, ONOFF, SETPOINT

_COLORS = {"IT2FLS": FUZZY, "ON/OFF": ONOFF}


def plot_temperature_comparison(runs: Dict[str, pd.DataFrame], comfort_band: float = 1.0, figsize=(8.5, 3.8)):
    fig, ax = plt.subplots(figsize=figsize)
    first = next(iter(runs.values()))
    t_set = float(first["t_set"].iloc[0])
    ax.axhspan(t_set - comfort_band, t_set + comfort_band, color=FUZZY, alpha=0.08, label=f"comfort zone ±{comfort_band:g} °C")
    for name, df in runs.items():
        ax.plot(df["time_min"], df["t_room"], lw=2, color=_COLORS.get(name), label=name)
    ax.axhline(t_set, color=SETPOINT, ls="--", lw=1.2, label=f"T_set = {t_set:g} °C")
    ax.set_xlabel("Time [min]"); ax.set_ylabel("Temperature [°C]"); ax.set_title("Temperature: IT2FLS vs ON/OFF")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_control_comparison(runs: Dict[str, pd.DataFrame], figsize=(8.5, 3.6)):
    fig, axes = plt.subplots(len(runs), 1, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (name, df) in zip(axes, runs.items()):
        ax.step(df["time_min"], df["u"], where="post", color=_COLORS.get(name), lw=1.6)
        ax.fill_between(df["time_min"], 0, df["u"], step="post", color=_COLORS.get(name), alpha=0.25)
        ax.axhline(0, color=NEUTRAL, lw=0.8)
        ax.set_ylim(-105, 105); ax.set_ylabel(f"{name}\nu")
    axes[0].set_title("Control signals (− heating | + cooling)")
    axes[-1].set_xlabel("Time [min]")
    fig.tight_layout()
    return fig


def plot_fou_impact(curves: Dict[float, dict], figsize=(8, 3.8)):
    """Wpływ szerokości FOU na odpowiedź u(e) i szerokość przedziału typu zredukowanego."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0] + 2, figsize[1]), gridspec_kw={"width_ratios": [1.3, 1]})
    cmap = plt.get_cmap("viridis")
    keys = sorted(curves)
    for i, f in enumerate(keys):
        c = cmap(i / max(1, len(keys) - 1))
        ax1.plot(curves[f]["e"], curves[f]["crisp"], color=c, lw=1.8, label=f"FOU factor {f:g}")
    ax1.axhline(0, color=NEUTRAL, lw=0.8)
    ax1.set_xlabel("Temperature error e [°C]"); ax1.set_ylabel("Crisp output u")
    ax1.set_title("Crisp response vs. FOU width"); ax1.legend(loc="upper left")
    for i, f in enumerate(keys):
        c = cmap(i / max(1, len(keys) - 1))
        ax2.plot(curves[f]["e"], curves[f]["y_right"] - curves[f]["y_left"], color=c, lw=1.8)
    ax2.set_xlabel("Temperature error e [°C]"); ax2.set_ylabel("Width of [y_left, y_right]")
    ax2.set_title("Type-reduced interval width")
    fig.tight_layout()
    return fig
