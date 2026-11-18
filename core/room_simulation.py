"""Uproszczony model termiczny pomieszczenia (model demonstracyjny, czas w minutach).

Równanie ciągłe (jedna strefa, bezwładność pierwszego rzędu):

    dT/dt = (T_out - T)/tau + P(u)
    P(u)  = heat_rate * max(-u, 0)/100  -  cool_rate * max(u, 0)/100        [°C/min]

u < 0 -> ogrzewanie (P > 0, temperatura rośnie), u > 0 -> chłodzenie (P < 0).
Przy stałym u w kroku dt rozwiązanie jest dokładne (całkowanie wykładnicze,
bezwarunkowo stabilne, także dla dużych dt):

    T_eq   = T_out + tau * P(u)
    T(t+dt) = T_eq + (T(t) - T_eq) * exp(-dt/tau) [+ zakłócenie]
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Protocol

import numpy as np
import pandas as pd

from config import default_config as cfg


class SupportsControl(Protocol):
    def control(self, t_room: float, t_set: float, humidity: float) -> float: ...


@dataclass(frozen=True)
class RoomParams:
    """Parametry symulacji (walidowane względem ``SIM_LIMITS``)."""

    t_out: float = cfg.SIM_DEFAULTS["t_out"]
    t0: float = cfg.SIM_DEFAULTS["t0"]
    t_set: float = cfg.SIM_DEFAULTS["t_set"]
    humidity: float = cfg.SIM_DEFAULTS["humidity"]
    duration_min: float = cfg.SIM_DEFAULTS["duration_min"]
    dt_min: float = cfg.SIM_DEFAULTS["dt_min"]
    tau_min: float = cfg.SIM_DEFAULTS["tau_min"]
    heat_rate: float = cfg.SIM_DEFAULTS["heat_rate"]
    cool_rate: float = cfg.SIM_DEFAULTS["cool_rate"]
    noise_std: float = cfg.SIM_DEFAULTS["noise_std"]
    seed: int = cfg.SIM_DEFAULTS["seed"]

    def __post_init__(self) -> None:
        for key, (lo, hi) in cfg.SIM_LIMITS.items():
            if hasattr(self, key):
                v = getattr(self, key)
                if not (np.isfinite(v) and lo <= v <= hi):
                    raise ValueError(f"Parametr {key}={v} poza zakresem [{lo}, {hi}]")
        if self.n_steps > cfg.MAX_SIM_STEPS:
            raise ValueError(f"Zbyt wiele kroków symulacji ({self.n_steps} > {cfg.MAX_SIM_STEPS})")

    @property
    def n_steps(self) -> int:
        return int(round(self.duration_min / self.dt_min))

    @classmethod
    def clamped(cls, **kwargs) -> "RoomParams":
        """Tworzy parametry, ograniczając wartości do dozwolonych zakresów."""
        out = {}
        for key, v in kwargs.items():
            if key in cfg.SIM_LIMITS:
                lo, hi = cfg.SIM_LIMITS[key]
                v = float(np.clip(v, lo, hi))
            out[key] = v
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in out.items() if k in names})


def heating_power(u: float, p: RoomParams) -> float:
    """Moc cieplna netto P(u) w °C/min (dodatnia = ogrzewanie)."""
    u = float(np.clip(u, *cfg.OUTPUT_RANGE))
    return p.heat_rate * max(-u, 0.0) / 100.0 - p.cool_rate * max(u, 0.0) / 100.0


def room_step(t_room: float, u: float, p: RoomParams, noise: float = 0.0) -> float:
    """Jeden krok modelu pomieszczenia (całkowanie wykładnicze)."""
    t_eq = p.t_out + p.tau_min * heating_power(u, p)
    t_next = t_eq + (t_room - t_eq) * math.exp(-p.dt_min / p.tau_min) + noise
    return float(np.clip(t_next, -50.0, 80.0))


@dataclass
class SimulationResult:
    name: str
    params: RoomParams
    data: pd.DataFrame  # kolumny: time_min, t_room, t_set, error, u, heating, cooling


def simulate(controller: SupportsControl, params: RoomParams, name: str = "controller") -> SimulationResult:
    """Symuluje pętlę zamkniętą: T_k -> u_k = controller(T_k) -> T_{k+1}."""
    if hasattr(controller, "reset"):
        controller.reset()
    n = params.n_steps
    rng = np.random.default_rng(params.seed)
    noise = rng.normal(0.0, params.noise_std, n) if params.noise_std > 0 else np.zeros(n)
    t = np.empty(n + 1)
    u = np.empty(n + 1)
    t[0] = params.t0
    for k in range(n + 1):
        u[k] = float(np.clip(controller.control(t[k], params.t_set, params.humidity), *cfg.OUTPUT_RANGE))
        if k < n:
            t[k + 1] = room_step(t[k], u[k], params, noise[k])
    df = pd.DataFrame({
        "time_min": np.arange(n + 1) * params.dt_min,
        "t_room": t,
        "t_set": params.t_set,
        "error": t - params.t_set,
        "u": u,
        "heating": np.clip(-u, 0, None),
        "cooling": np.clip(u, 0, None),
    })
    return SimulationResult(name, params, df)
