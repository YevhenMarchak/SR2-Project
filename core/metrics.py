"""Metryki porównawcze regulatorów (liczone z danych symulacji).

Uwaga: ``energy_proxy`` to *uproszczony wskaźnik intensywności sterowania*,
a nie zużycie energii w kWh -- model nie jest skalibrowany fizycznie.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def mae(error) -> float:
    """Mean Absolute Error: mean(|e|)."""
    e = np.asarray(error, dtype=float)
    return float(np.mean(np.abs(e)))


def rmse(error) -> float:
    """Root Mean Squared Error: sqrt(mean(e^2))."""
    e = np.asarray(error, dtype=float)
    return float(np.sqrt(np.mean(e ** 2)))


def time_to_comfort(time, error, band: float) -> Optional[float]:
    """Pierwszy moment, w którym |e| <= band (None, jeśli nigdy)."""
    time, error = np.asarray(time, float), np.asarray(error, float)
    idx = np.flatnonzero(np.abs(error) <= band)
    return float(time[idx[0]]) if idx.size else None


def settling_time(time, error, band: float) -> Optional[float]:
    """Czas, od którego |e| <= band już do końca symulacji (None, jeśli nie ustaliło się)."""
    time, error = np.asarray(time, float), np.asarray(error, float)
    outside = np.flatnonzero(np.abs(error) > band)
    if outside.size == 0:
        return float(time[0])
    last = outside[-1]
    return float(time[last + 1]) if last + 1 < time.size else None


def overshoot_undershoot(t_room, t_set: float):
    """(overshoot, undershoot) w °C względem kierunku dojścia do wartości zadanej.

    overshoot  -- maksymalne przekroczenie T_set w kierunku dojścia;
    undershoot -- maksymalny spadek po stronie przeciwnej po pierwszym osiągnięciu T_set.
    Gdy T(0) == T_set, overshoot = max(T - T_set)+, undershoot = max(T_set - T)+.
    """
    t = np.asarray(t_room, float)
    d = np.sign(t_set - t[0])
    if d > 0:
        reached = np.flatnonzero(t >= t_set)
        over = max(0.0, t.max() - t_set)
        under = max(0.0, t_set - t[reached[0]:].min()) if reached.size else 0.0
    elif d < 0:
        reached = np.flatnonzero(t <= t_set)
        over = max(0.0, t_set - t.min())
        under = max(0.0, t[reached[0]:].max() - t_set) if reached.size else 0.0
    else:
        over, under = max(0.0, t.max() - t_set), max(0.0, t_set - t.min())
    return float(over), float(under)


def count_control_changes(u, threshold: float = 5.0) -> int:
    """Liczba znaczących zmian sterowania: kroków, w których |u_k - u_{k-1}| > threshold."""
    u = np.asarray(u, float)
    return int(np.sum(np.abs(np.diff(u)) > threshold))


def energy_proxy(u, dt_min: float) -> float:
    """Wskaźnik intensywności sterowania: sum(|u|/100 * dt) [minuty pełnej mocy]."""
    u = np.asarray(u, float)
    return float(np.sum(np.abs(u) / 100.0) * dt_min)


def compute_metrics(df: pd.DataFrame, dt_min: float, comfort_band: float = 1.0,
                    change_threshold: float = 5.0) -> Dict[str, Optional[float]]:
    """Komplet metryk dla wyniku symulacji (``SimulationResult.data``)."""
    over, under = overshoot_undershoot(df["t_room"].values, float(df["t_set"].iloc[0]))
    return {
        "MAE [°C]": mae(df["error"]),
        "RMSE [°C]": rmse(df["error"]),
        "Time to comfort [min]": time_to_comfort(df["time_min"], df["error"], comfort_band),
        "Settling time [min]": settling_time(df["time_min"], df["error"], comfort_band),
        "Overshoot [°C]": over,
        "Undershoot [°C]": under,
        "Control changes [-]": count_control_changes(df["u"], change_threshold),
        "Energy proxy [full-power min]": energy_proxy(df["u"], dt_min),
    }
