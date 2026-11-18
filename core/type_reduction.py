"""Redukcja typu Karnika-Mendela (KM) dla przedziałowych systemów rozmytych typu 2.

Model: *center-of-sets* type reduction. Każda reguła l ma konsekwencję w postaci
zbioru (typu 2), reprezentowanego przez swój przedział centroidu [c_l, c_r],
oraz przedział siły odpalenia [f_l, F_l]. Zredukowany zbiór ma postać przedziału
[y_left, y_right]:

    y_left  = min_{f_i in [f_l, F_l]}  sum(f_i * c_l_i) / sum(f_i)
    y_right = max_{f_i in [f_l, F_l]}  sum(f_i * c_r_i) / sum(f_i)

Ekstrema leżą na brzegach przedziałów firing strength, a ich położenie
wyznacza punkt przełączenia (switch point) k. Algorytm KM znajduje go
iteracyjnie (zbieżność w co najwyżej N krokach).

Ten sam algorytm (z y = x_i, f = [LMF(x_i), UMF(x_i)]) służy do wyznaczania
centroidu pojedynczego zbioru IT2 (``it2_centroid``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class KMEndpoint:
    """Wynik jednego końca algorytmu KM."""

    value: float
    switch_point: int   # k (1-based) w posortowanej liście y; 0 gdy N == 1
    iterations: int
    converged: bool


@dataclass(frozen=True)
class TypeReductionResult:
    """Przedział typu zredukowanego [y_left, y_right] i wartość crisp."""

    y_left: float
    y_right: float
    left: KMEndpoint
    right: KMEndpoint

    @property
    def crisp(self) -> float:
        """Defuzzyfikacja: środek przedziału, (y_left + y_right) / 2."""
        return 0.5 * (self.y_left + self.y_right)

    @property
    def width(self) -> float:
        return self.y_right - self.y_left


def _validate(y, f_lower, f_upper, eps: float = 1e-12):
    y = np.asarray(y, dtype=float).ravel()
    fl = np.asarray(f_lower, dtype=float).ravel()
    fu = np.asarray(f_upper, dtype=float).ravel()
    if not (y.shape == fl.shape == fu.shape):
        raise ValueError("y, f_lower i f_upper muszą mieć ten sam kształt")
    if y.size == 0:
        raise ValueError("Puste dane wejściowe type reduction")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(fl)) and np.all(np.isfinite(fu))):
        raise ValueError("Dane type reduction zawierają NaN/inf")
    if np.any(fl < -eps) or np.any(fu < -eps):
        raise ValueError("Siły odpalenia muszą być >= 0")
    if np.any(fl > fu + eps):
        raise ValueError("Wymagane f_lower <= f_upper (niewłaściwy przedział)")
    fl = np.clip(fl, 0.0, None)
    fu = np.maximum(np.clip(fu, 0.0, None), fl)
    return y, fl, fu


def km_endpoint(y, f_lower, f_upper, side: str, tol: float = 1e-12) -> KMEndpoint:
    """Jeden koniec algorytmu KM.

    Args:
        y: położenia (centroidy konsekwencji), dowolna kolejność.
        f_lower, f_upper: dolne/górne siły odpalenia.
        side: ``"left"`` (minimum, y_left) lub ``"right"`` (maksimum, y_right).

    Kroki (Mendel 2001; Karnik & Mendel 2001):
      0. Posortuj y rosnąco; pomiń elementy z F_i = 0.
      1. Ustaw f_i = (f_l_i + F_l_i)/2, policz y' = sum(f y)/sum(f).
      2. Znajdź k: y_k <= y' <= y_{k+1}.
      3. Prawy koniec: f_i = f_l_i dla i <= k, F_i dla i > k;
         lewy koniec:  f_i = F_i dla i <= k, f_l_i dla i > k. Policz y''.
      4. Jeśli y'' == y' -- koniec; w przeciwnym razie y' <- y'' i wróć do 2.
    """
    if side not in ("left", "right"):
        raise ValueError("side musi być 'left' lub 'right'")
    y, fl, fu = _validate(y, f_lower, f_upper)
    keep = fu > 0.0
    if not np.any(keep):
        raise ValueError("Brak aktywnych elementów (wszystkie F_i = 0)")
    y, fl, fu = y[keep], fl[keep], fu[keep]
    order = np.argsort(y, kind="stable")
    y, fl, fu = y[order], fl[order], fu[order]
    n = y.size
    if n == 1:
        return KMEndpoint(float(y[0]), 0, 0, True)

    idx = np.arange(n)
    w = 0.5 * (fl + fu)
    y_prime = float(np.dot(w, y) / w.sum())
    max_iter = 2 * n + 10
    k = 0
    for it in range(1, max_iter + 1):
        k = int(np.clip(np.searchsorted(y, y_prime, side="right") - 1, 0, n - 2))
        if side == "right":
            w = np.where(idx <= k, fl, fu)
        else:
            w = np.where(idx <= k, fu, fl)
        y_new = float(np.dot(w, y) / w.sum())  # w.sum() > 0: element F_i > 0 po stronie "wysokiej"
        if abs(y_new - y_prime) <= tol * max(1.0, abs(y_prime)):
            return KMEndpoint(y_new, k + 1, it, True)
        y_prime = y_new
    return KMEndpoint(y_prime, k + 1, max_iter, False)


def km_endpoints(y, f_lower, f_upper) -> Tuple[KMEndpoint, KMEndpoint]:
    """(lewy, prawy) koniec KM dla wspólnego wektora y."""
    return km_endpoint(y, f_lower, f_upper, "left"), km_endpoint(y, f_lower, f_upper, "right")


def type_reduce_center_of_sets(c_left, c_right, f_lower, f_upper) -> TypeReductionResult:
    """Type reduction center-of-sets dla konsekwencji o przedziałach centroidów.

    y_left korzystają z lewych końców centroidów, y_right z prawych.
    Dla konsekwencji o zerowej niepewności (c_left == c_right) otrzymujemy
    klasyczne KM dla singletonów.
    """
    c_left = np.asarray(c_left, dtype=float)
    c_right = np.asarray(c_right, dtype=float)
    if np.any(c_left > c_right + 1e-9):
        raise ValueError("Centroid: wymagane c_left <= c_right")
    left = km_endpoint(c_left, f_lower, f_upper, "left")
    right = km_endpoint(c_right, f_lower, f_upper, "right")
    yl, yr = left.value, right.value
    if yl > yr:  # tylko błąd numeryczny rzędu 1e-12
        yl = yr = 0.5 * (yl + yr)
    return TypeReductionResult(yl, yr, left, right)


def it2_centroid(x, lmf, umf) -> Tuple[float, float]:
    """Przedział centroidu [c_l, c_r] dyskretnego zbioru IT2 (algorytm KM)."""
    left, right = km_endpoints(x, lmf, umf)
    return left.value, right.value
