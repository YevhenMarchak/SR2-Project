"""Zmienne lingwistyczne (temperatura, wilgotność, wyjście) z funkcjami IT2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from config import default_config as cfg
from core.membership_type2 import IT2TrapezoidMF, trapezoid


@dataclass(frozen=True)
class LinguisticVariable:
    """Zmienna lingwistyczna: nazwa, dziedzina i uporządkowana lista zbiorów IT2."""

    name: str
    universe: Tuple[float, float]
    unit: str
    terms: Tuple[IT2TrapezoidMF, ...]

    def __post_init__(self) -> None:
        lo, hi = self.universe
        if not lo < hi:
            raise ValueError("Dziedzina musi spełniać lo < hi")
        names = [t.name for t in self.terms]
        if len(set(names)) != len(names):
            raise ValueError(f"Zduplikowane nazwy zbiorów w zmiennej {self.name}")

    @property
    def term_names(self) -> List[str]:
        return [t.name for t in self.terms]

    def term(self, name: str) -> IT2TrapezoidMF:
        for t in self.terms:
            if t.name == name:
                return t
        raise KeyError(f"{self.name}: brak zbioru '{name}'")

    def clip(self, x):
        """Ogranicza wartość do dziedziny (stabilność poza zakresem modelu)."""
        return np.clip(x, self.universe[0], self.universe[1])

    def memberships(self, x) -> Tuple[np.ndarray, np.ndarray]:
        """(lower, upper) dla wszystkich zbiorów naraz.

        Dla skalara x zwraca tablice kształtu (n_terms,); dla tablicy x
        kształtu (m,) zwraca (n_terms, m).
        """
        x = np.asarray(self.clip(x), dtype=float)
        scalar = x.ndim == 0
        xv = x.reshape(1, -1) if not scalar else x.reshape(1, 1)
        p = np.array([t.params for t in self.terms])[:, :, None]
        s = np.array([t.uncertainty for t in self.terms])[:, None]
        a, b, c, d = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        upper = trapezoid(xv, a, b, c, d, -s, +s)
        lower = trapezoid(xv, a, b, c, d, +s, -s)
        if scalar:
            return lower[:, 0], upper[:, 0]
        return lower, upper

    def membership_dict(self, x) -> Dict[str, Tuple[float, float]]:
        lo, hi = self.memberships(float(x))
        return {n: (float(l), float(h)) for n, l, h in zip(self.term_names, lo, hi)}

    def grid(self, n: int = 1001) -> np.ndarray:
        return np.linspace(self.universe[0], self.universe[1], n)


def _build(name: str, universe, unit: str, terms: Dict[str, Sequence[float]], s: float) -> LinguisticVariable:
    return LinguisticVariable(
        name=name,
        universe=tuple(universe),
        unit=unit,
        terms=tuple(IT2TrapezoidMF(n, *p, uncertainty=s) for n, p in terms.items()),
    )


def _shift(kind: str, fou_factor: float) -> float:
    if not (np.isfinite(fou_factor) and fou_factor >= 0):
        raise ValueError("Współczynnik FOU musi być skończony i >= 0")
    lo, hi = cfg.FOU_FACTOR_RANGE
    return float(np.clip(fou_factor, lo, hi)) * cfg.FOU_SCALE[kind]


def build_temperature_variable(fou_factor: float = cfg.DEFAULT_FOU_FACTOR) -> LinguisticVariable:
    return _build("Temperature Error", cfg.TEMP_ERROR_RANGE, "°C", cfg.TEMPERATURE_TERMS,
                  _shift("temperature", fou_factor))


def build_humidity_variable(fou_factor: float = cfg.DEFAULT_FOU_FACTOR) -> LinguisticVariable:
    return _build("Humidity", cfg.HUMIDITY_RANGE, "%", cfg.HUMIDITY_TERMS, _shift("humidity", fou_factor))


def build_output_variable(fou_factor: float = cfg.DEFAULT_FOU_FACTOR) -> LinguisticVariable:
    return _build("Control Output", cfg.OUTPUT_RANGE, "", cfg.OUTPUT_TERMS, _shift("output", fou_factor))


def build_variables(fou_factor: float = cfg.DEFAULT_FOU_FACTOR):
    """(temperatura, wilgotność, wyjście) dla wspólnego współczynnika FOU."""
    return (build_temperature_variable(fou_factor), build_humidity_variable(fou_factor),
            build_output_variable(fou_factor))
