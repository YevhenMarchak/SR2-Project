"""Sterownik IT2FLS: łączy fuzzyfikację, wnioskowanie i redukcję typu KM."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import default_config as cfg
from core.inference import IT2MamdaniEngine, RuleActivation
from core.type_reduction import TypeReductionResult, type_reduce_center_of_sets
from core.variables import build_variables

HEATING, NEUTRAL, COOLING = "HEATING", "NEUTRAL", "COOLING"


def classify_state(u: float, threshold: float = cfg.NEUTRAL_THRESHOLD) -> str:
    """u < 0 -> ogrzewanie, u > 0 -> chłodzenie, |u| < threshold -> neutralnie."""
    if abs(u) < threshold:
        return NEUTRAL
    return HEATING if u < 0 else COOLING


@dataclass
class ControllerResult:
    """Wynik jednego wywołania regulatora wraz z danymi diagnostycznymi."""

    t_room: float
    t_set: float
    humidity: float
    temp_error_raw: float             # e = T_room - T_set (przed ograniczeniem)
    temp_error: float                 # e po ograniczeniu do dziedziny modelu
    temp_memberships: Dict[str, Tuple[float, float]]
    humidity_memberships: Dict[str, Tuple[float, float]]
    activations: List[RuleActivation]
    f_lower: np.ndarray
    f_upper: np.ndarray
    type_reduction: Optional[TypeReductionResult]
    y_left: float
    y_right: float
    crisp: float                      # wynik końcowy w [-100, 100]
    state: str
    no_rule_fired: bool = False
    aggregated: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    clipped: list = field(default_factory=list)

    @property
    def active_rules(self) -> List[RuleActivation]:
        """Aktywne reguły posortowane malejąco według górnej siły odpalenia."""
        return sorted([a for a in self.activations if a.active], key=lambda a: -a.f_upper)

    @property
    def intensity_percent(self) -> float:
        return abs(self.crisp)


class IT2FuzzyController:
    """Regulator HVAC oparty na interval Type-2 FLS (Mamdani + KM)."""

    def __init__(self, fou_factor: float = cfg.DEFAULT_FOU_FACTOR,
                 neutral_threshold: float = cfg.NEUTRAL_THRESHOLD) -> None:
        self.fou_factor = float(fou_factor)
        self.neutral_threshold = float(neutral_threshold)
        self.temperature, self.humidity, self.output = build_variables(self.fou_factor)
        self.engine = IT2MamdaniEngine(self.temperature, self.humidity, self.output)

    def compute(self, t_room: float, t_set: float, humidity: float, detailed: bool = True) -> ControllerResult:
        """Pełny przebieg: błąd -> fuzzyfikacja -> reguły -> agregacja -> KM -> crisp."""
        for name, v in (("t_room", t_room), ("t_set", t_set), ("humidity", humidity)):
            if not np.isfinite(v):
                raise ValueError(f"{name} musi być liczbą skończoną")
        e_raw = float(t_room) - float(t_set)
        e = float(self.temperature.clip(e_raw))
        h = float(self.humidity.clip(humidity))

        f_lo, f_hi = self.engine.fire(e, h)
        no_rule = not np.any(f_hi > 0)
        if no_rule:  # nie powinno się zdarzyć przy pełnym pokryciu; zachowanie bezpieczne
            tr, yl, yr, crisp = None, 0.0, 0.0, 0.0
        else:
            c_l, c_r = self.engine.rule_centroids
            tr = type_reduce_center_of_sets(c_l, c_r, f_lo, f_hi)
            yl, yr = tr.y_left, tr.y_right
            lo, hi = cfg.OUTPUT_RANGE
            crisp = float(np.clip(tr.crisp, lo, hi))

        agg, clipped, t_m, h_m, acts = None, [], {}, {}, []
        if detailed:
            t_m, h_m = self.temperature.membership_dict(e), self.humidity.membership_dict(h)
            acts = self.engine.activations(f_lo, f_hi)
            if not no_rule:
                agg = self.engine.aggregate(f_lo, f_hi)
                clipped = self.engine.clipped_consequents(f_lo, f_hi)
        return ControllerResult(
            t_room=float(t_room), t_set=float(t_set), humidity=h, temp_error_raw=e_raw, temp_error=e,
            temp_memberships=t_m, humidity_memberships=h_m, activations=acts,
            f_lower=f_lo, f_upper=f_hi, type_reduction=tr, y_left=yl, y_right=yr, crisp=crisp,
            state=classify_state(crisp, self.neutral_threshold), no_rule_fired=no_rule,
            aggregated=agg, clipped=clipped,
        )

    def control(self, t_room: float, t_set: float, humidity: float) -> float:
        """Szybka ścieżka (bez diagnostyki) -- tylko wartość sterująca u."""
        return self.compute(t_room, t_set, humidity, detailed=False).crisp
