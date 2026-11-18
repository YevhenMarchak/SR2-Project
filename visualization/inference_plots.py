"""Wykresy inferencji: odpowiedź regulatora, heatmapa, reguły, agregacja i type reduction."""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from core.controller import ControllerResult
from visualization.style import COOL, FUZZY, HEAT, NEUTRAL, PALETTE


def plot_response_curve(curve: dict, e_current: Optional[float] = None, humidity: Optional[float] = None,
                        figsize=(7, 3.8)):
    """u(e) przy stałej wilgotności: crisp + przedział typu zredukowanego [y_left, y_right]."""
    fig, ax = plt.subplots(figsize=figsize)
    e = curve["e"]
    ax.fill_between(e, curve["y_left"], curve["y_right"], color=FUZZY, alpha=0.25, label="type-reduced interval [y_left, y_right]")
    ax.plot(e, curve["crisp"], color=FUZZY, lw=2.2, label="crisp output u")
    ax.axhline(0, color=NEUTRAL, lw=0.8)
    ax.axvline(0, color=NEUTRAL, lw=0.8, ls=":")
    if e_current is not None:
        ec = float(np.clip(e_current, e.min(), e.max()))
        uc = float(np.interp(ec, e, curve["crisp"]))
        ax.axvline(ec, color="#222", ls=":", lw=1.2)
        ax.plot([ec], [uc], "o", color="#222", ms=7, label=f"current: e={ec:+.1f}, u={uc:+.1f}")
    ax.set_ylim(-105, 105)
    ax.set_xlabel("Temperature error e = T_room − T_set [°C]")
    ax.set_ylabel("Control output u")
    ax.set_title("Controller response" + (f" (humidity {humidity:.0f}%)" if humidity is not None else ""))
    ax.text(e.min(), -98, "heating", color=HEAT, fontsize=9, va="bottom")
    ax.text(e.min(), 92, "cooling", color=COOL, fontsize=9, va="top")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_heatmap(e_grid, h_grid, z, current: Optional[tuple] = None, figsize=(7, 4.2)):
    """Powierzchnia sterowania u(e, h) policzona regulatorem na siatce."""
    fig, ax = plt.subplots(figsize=figsize)
    mesh = ax.pcolormesh(e_grid, h_grid, z, cmap="coolwarm_r", norm=Normalize(-100, 100), shading="gouraud")
    cs = ax.contour(e_grid, h_grid, z, levels=[-60, -30, -5, 5, 30, 60], colors="k", linewidths=0.5, alpha=0.4)
    ax.clabel(cs, fmt="%d", fontsize=7)
    if current is not None:
        ax.plot([np.clip(current[0], e_grid.min(), e_grid.max())], [np.clip(current[1], h_grid.min(), h_grid.max())],
                "o", ms=9, mfc="white", mec="#111", mew=2, label="current state")
        ax.legend(loc="upper left")
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label("Control output u  (− heating | + cooling)")
    ax.set_xlabel("Temperature error e [°C]")
    ax.set_ylabel("Relative humidity [%]")
    ax.set_title("Control surface of the IT2FLS")
    ax.grid(False)
    fig.tight_layout()
    return fig


def plot_firing_strengths(result: ControllerResult, figsize=(7, 3.6)):
    """Przedziały siły odpalenia [f_lower, f_upper] aktywnych reguł."""
    acts = result.active_rules
    fig, ax = plt.subplots(figsize=figsize)
    if not acts:
        ax.text(0.5, 0.5, "No active rules", ha="center", transform=ax.transAxes)
        return fig
    y = np.arange(len(acts))[::-1]
    for yi, a in zip(y, acts):
        ax.barh(yi, a.f_upper, color=FUZZY, alpha=0.30, height=0.6)
        ax.barh(yi, a.f_lower, color=FUZZY, alpha=0.90, height=0.6)
        ax.text(a.f_upper + 0.01, yi, f"[{a.f_lower:.2f}, {a.f_upper:.2f}]", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"R{a.rule.rule_id}: {a.rule.temperature_term} & {a.rule.humidity_term} → {a.rule.output_term}"
                        for a in acts], fontsize=8)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Firing strength (dark = lower f, light = upper F)")
    ax.set_title("Active rules – interval firing strengths")
    fig.tight_layout()
    return fig


def plot_aggregation(result: ControllerResult, figsize=(8, 4.4)):
    """Zagregowany przedziałowy zbiór wyjściowy oraz wynik type reduction [y_left, y_right]."""
    fig, ax = plt.subplots(figsize=figsize)
    if result.aggregated is None:
        ax.text(0.5, 0.5, "No aggregated set", ha="center", transform=ax.transAxes)
        return fig
    x, lo, hi = result.aggregated
    for i, (rule, c_lo, c_hi) in enumerate(result.clipped):
        ax.plot(x, c_hi, color=PALETTE[i % len(PALETTE)], lw=0.9, alpha=0.6)
    ax.fill_between(x, lo, hi, color=FUZZY, alpha=0.30, label="aggregated FOU")
    ax.plot(x, hi, color=FUZZY, lw=2, label="aggregated UMF")
    ax.plot(x, lo, color=FUZZY, lw=1.3, ls="--", label="aggregated LMF")
    ax.axvspan(result.y_left, result.y_right, color="#f2c14e", alpha=0.35, label=f"[y_left, y_right] = [{result.y_left:.1f}, {result.y_right:.1f}]")
    ax.axvline(result.crisp, color="#222", lw=2, label=f"crisp u = {result.crisp:+.1f}")
    ax.set_xlim(-100, 100)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Control output u")
    ax.set_ylabel("Membership degree")
    ax.set_title("Aggregation and Karnik–Mendel type reduction")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    return fig
