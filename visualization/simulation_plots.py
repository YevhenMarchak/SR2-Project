"""Wykresy symulacji pojedynczego regulatora."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from visualization.style import COOL, FUZZY, HEAT, NEUTRAL, SETPOINT


def plot_temperature(df: pd.DataFrame, comfort_band: float = 1.0, title: str = "Room temperature",
                     color: str = FUZZY, figsize=(8, 3.6)):
    fig, ax = plt.subplots(figsize=figsize)
    t_set = float(df["t_set"].iloc[0])
    ax.axhspan(t_set - comfort_band, t_set + comfort_band, color=FUZZY, alpha=0.10, label=f"comfort zone ±{comfort_band:g} °C")
    ax.plot(df["time_min"], df["t_room"], color=color, lw=2.2, label="T_room")
    ax.axhline(t_set, color=SETPOINT, ls="--", lw=1.3, label=f"T_set = {t_set:g} °C")
    ax.set_xlabel("Time [min]"); ax.set_ylabel("Temperature [°C]"); ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_control(df: pd.DataFrame, title: str = "Control signal", figsize=(8, 3.0)):
    fig, ax = plt.subplots(figsize=figsize)
    t, u = df["time_min"], df["u"]
    ax.fill_between(t, 0, u.clip(upper=0), color=HEAT, alpha=0.45, step="post", label="heating (u < 0)")
    ax.fill_between(t, 0, u.clip(lower=0), color=COOL, alpha=0.45, step="post", label="cooling (u > 0)")
    ax.step(t, u, where="post", color="#333", lw=1.1)
    ax.axhline(0, color=NEUTRAL, lw=1)
    ax.set_ylim(-105, 105)
    ax.set_xlabel("Time [min]"); ax.set_ylabel("Control output u"); ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig
