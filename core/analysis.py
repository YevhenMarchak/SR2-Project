"""Obliczenia pomocnicze do wykresów: wszystkie dane pochodzą z silnika IT2FLS."""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from core.controller import IT2FuzzyController


def response_curve(controller: IT2FuzzyController, e_grid: np.ndarray, humidity: float,
                   t_set: float = 22.0) -> Dict[str, np.ndarray]:
    """Odpowiedź regulatora na błąd temperatury przy stałej wilgotności.

    Zwraca crisp oraz przedział typu zredukowanego [y_left, y_right].
    """
    crisp, yl, yr = [], [], []
    for e in e_grid:
        r = controller.compute(t_set + float(e), t_set, humidity, detailed=False)
        crisp.append(r.crisp); yl.append(r.y_left); yr.append(r.y_right)
    return {"e": np.asarray(e_grid), "crisp": np.array(crisp), "y_left": np.array(yl), "y_right": np.array(yr)}


def control_surface(controller: IT2FuzzyController, e_grid: np.ndarray, h_grid: np.ndarray,
                    t_set: float = 22.0) -> np.ndarray:
    """Siatka wyników sterowania Z[i_h, i_e] = u(e_grid[i_e], h_grid[i_h])."""
    z = np.empty((len(h_grid), len(e_grid)))
    for i, h in enumerate(h_grid):
        for j, e in enumerate(e_grid):
            z[i, j] = controller.compute(t_set + float(e), t_set, float(h), detailed=False).crisp
    return z


def fou_impact(factors: Sequence[float], e_grid: np.ndarray, humidity: float) -> Dict[float, Dict[str, np.ndarray]]:
    """Odpowiedź regulatora dla różnych szerokości niepewności (FOU)."""
    return {float(f): response_curve(IT2FuzzyController(f), e_grid, humidity) for f in factors}
