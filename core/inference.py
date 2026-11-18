"""Silnik wnioskowania Mamdaniego dla przedziałowego systemu rozmytego typu 2.

Operatory: T-norm = minimum (spójnik AND oraz implikacja), S-norm = maksimum
(agregacja reguł). Dla wejść x = (e, h) i reguły l:

    f_l(x) = min( LMF_Tl(e), LMF_Hl(h) )    -- dolna siła odpalenia
    F_l(x) = min( UMF_Tl(e), UMF_Hl(h) )    -- górna siła odpalenia

Zagregowany przedziałowy zbiór wyjściowy (do wizualizacji):

    UMF_agg(y) = max_l min( F_l, UMF_Bl(y) ),   LMF_agg(y) = max_l min( f_l, LMF_Bl(y) )

Redukcja typu (``type_reduction``) pracuje na przedziałach [f_l, F_l] i
centroidach zbiorów konsekwencji (center-of-sets), co jest równoważne
defuzzyfikacji zagregowanego zbioru Mamdaniego w sensie przedziałowym.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import default_config as cfg
from core.rule_base import RULES, Rule
from core.type_reduction import it2_centroid
from core.variables import LinguisticVariable


@dataclass(frozen=True)
class RuleActivation:
    """Aktywacja reguły: przedział siły odpalenia [f_lower, f_upper]."""

    rule: Rule
    f_lower: float
    f_upper: float

    @property
    def active(self) -> bool:
        return self.f_upper > 0.0

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.f_lower + self.f_upper)


class IT2MamdaniEngine:
    """Fuzzyfikacja, odpalanie reguł i agregacja (bez redukcji typu)."""

    def __init__(self, temperature: LinguisticVariable, humidity: LinguisticVariable,
                 output: LinguisticVariable, rules: Optional[Sequence[Rule]] = None,
                 grid_points: int = cfg.OUTPUT_GRID_POINTS) -> None:
        self.temperature, self.humidity, self.output = temperature, humidity, output
        self.rules: List[Rule] = list(RULES if rules is None else rules)
        if not self.rules:
            raise ValueError("Baza reguł jest pusta")
        t_names, h_names, o_names = temperature.term_names, humidity.term_names, output.term_names
        try:
            self._t_idx = np.array([t_names.index(r.temperature_term) for r in self.rules])
            self._h_idx = np.array([h_names.index(r.humidity_term) for r in self.rules])
            self._o_idx = np.array([o_names.index(r.output_term) for r in self.rules])
        except ValueError as exc:
            raise ValueError(f"Reguła odwołuje się do nieistniejącego zbioru: {exc}") from exc
        self.output_grid = output.grid(grid_points)
        self._out_lmf, self._out_umf = output.memberships(self.output_grid)
        cents = [it2_centroid(self.output_grid, self._out_lmf[i], self._out_umf[i])
                 for i in range(len(o_names))]
        self.centroid_left = np.array([c[0] for c in cents])
        self.centroid_right = np.array([c[1] for c in cents])

    # -- krok 1-2: fuzzyfikacja i odpalenie reguł ------------------------------
    def fire(self, temp_error: float, humidity: float) -> Tuple[np.ndarray, np.ndarray]:
        """Przedziałowe siły odpalenia (f_lower, f_upper) dla wszystkich reguł."""
        t_lo, t_hi = self.temperature.memberships(temp_error)
        h_lo, h_hi = self.humidity.memberships(humidity)
        f_lo = np.minimum(t_lo[self._t_idx], h_lo[self._h_idx])
        f_hi = np.minimum(t_hi[self._t_idx], h_hi[self._h_idx])
        return f_lo, f_hi

    def activations(self, f_lower: np.ndarray, f_upper: np.ndarray) -> List[RuleActivation]:
        return [RuleActivation(r, float(l), float(u)) for r, l, u in zip(self.rules, f_lower, f_upper)]

    # -- agregacja (wizualizacja zbioru wyjściowego) --------------------------
    def aggregate(self, f_lower: np.ndarray, f_upper: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Zagregowane (x, LMF, UMF) zbioru wyjściowego; max po regułach z obcięciem min."""
        lower = np.zeros_like(self.output_grid)
        upper = np.zeros_like(self.output_grid)
        for l, fl, fu in zip(self._o_idx, f_lower, f_upper):
            if fu > 0:
                upper = np.maximum(upper, np.minimum(fu, self._out_umf[l]))
                lower = np.maximum(lower, np.minimum(fl, self._out_lmf[l]))
        return self.output_grid, lower, upper

    def clipped_consequents(self, f_lower, f_upper) -> List[Tuple[Rule, np.ndarray, np.ndarray]]:
        """Obcięte zbiory konsekwencji poszczególnych aktywnych reguł (do wykresów)."""
        out = []
        for r, l, fl, fu in zip(self.rules, self._o_idx, f_lower, f_upper):
            if fu > 0:
                out.append((r, np.minimum(fl, self._out_lmf[l]), np.minimum(fu, self._out_umf[l])))
        return out

    @property
    def rule_centroids(self) -> Tuple[np.ndarray, np.ndarray]:
        """Przedziały centroidów konsekwencji poszczególnych reguł (c_left, c_right)."""
        return self.centroid_left[self._o_idx], self.centroid_right[self._o_idx]
