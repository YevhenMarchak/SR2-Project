"""Przedziałowe (interval) funkcje przynależności typu 2.

Zbiór IT2 jest opisany przez parę funkcji UMF i LMF takich, że
``LMF(x) <= UMF(x)`` dla każdego x. Obszar między nimi to Footprint of
Uncertainty (FOU).

Konstrukcja (parametr niepewności ``s >= 0`` w jednostkach dziedziny):

    Nominalny trapez (a, b, c, d):
        T(x) = max(0, min(1, (x-a)/(b-a), (d-x)/(d-c)))

    UMF -- krawędzie rozsunięte o s:  rosnąca krawędź w lewo, opadająca w prawo
        UMF(x) = max(0, min(1, (x-(a-s))/(b-a), ((d+s)-x)/(d-c)))
    LMF -- krawędzie zsunięte o s:    rosnąca krawędź w prawo, opadająca w lewo
        LMF(x) = max(0, min(1, (x-(a+s))/(b-a), ((d-s)-x)/(d-c)))

Nachylenia zboczy są zachowane, więc każdy składnik minimum w LMF jest nie
większy niż odpowiadający mu składnik w UMF -- zatem LMF <= UMF *z konstrukcji*
(dla s = 0 oba są równe i dostajemy zbiór typu 1). W szczytach trójkątów LMF
ma niższą wysokość niż UMF, co daje niezerowe FOU także w rdzeniu zbioru.

Wartości ``-inf``/``+inf`` w (a, b) lub (c, d) oznaczają zbiór ramieniowy
(shoulder) bez odpowiedniego zbocza.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple

import numpy as np

ArrayLike = "float | np.ndarray"


def trapezoid(x, a, b, c, d, rise_shift=0.0, fall_shift=0.0):
    """Trapez z opcjonalnym przesunięciem zboczy; działa na skalarach i tablicach.

    Parametry mogą być tablicami (broadcasting), co pozwala policzyć wiele
    zbiorów naraz. Zbocze rosnące jest nieobecne, gdy ``a`` jest nieskończone,
    zbocze opadające -- gdy nieskończone jest ``d``.
    """
    x = np.asarray(x, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)
    fa, fd = np.isfinite(a), np.isfinite(d)
    a_s, b_s = np.where(fa, a, 0.0), np.where(fa, b, 1.0)
    c_s, d_s = np.where(fd, c, 0.0), np.where(fd, d, 1.0)
    rise = np.where(fa, (x - (a_s + rise_shift)) / (b_s - a_s), 1.0)
    fall = np.where(fd, ((d_s + fall_shift) - x) / (d_s - c_s), 1.0)
    return np.clip(np.minimum(np.minimum(rise, fall), 1.0), 0.0, 1.0)


@dataclass(frozen=True)
class IT2TrapezoidMF:
    """Przedziałowy zbiór rozmyty typu 2 oparty na trapezie (a, b, c, d)."""

    name: str
    a: float
    b: float
    c: float
    d: float
    uncertainty: float = 0.0  # s >= 0, przesunięcie krawędzi w jednostkach dziedziny

    def __post_init__(self) -> None:
        a, b, c, d = self.a, self.b, self.c, self.d
        if any(np.isnan(v) for v in (a, b, c, d)):
            raise ValueError(f"{self.name}: parametry nie mogą być NaN")
        if not (a <= b <= c <= d):
            raise ValueError(f"{self.name}: wymagane a <= b <= c <= d, dostano {(a, b, c, d)}")
        if np.isfinite(a) and not b > a:
            raise ValueError(f"{self.name}: skończone zbocze rosnące wymaga b > a")
        if np.isfinite(d) and not d > c:
            raise ValueError(f"{self.name}: skończone zbocze opadające wymaga d > c")
        # zbiór ramieniowy: nieskończoność dozwolona tylko jako a == b == -inf lub c == d == +inf
        if np.isinf(b) and not (a == b == -np.inf):
            raise ValueError(f"{self.name}: nieskończone b wymaga a == b == -inf")
        if np.isinf(c) and not (c == d == np.inf):
            raise ValueError(f"{self.name}: nieskończone c wymaga c == d == +inf")
        if not (np.isfinite(self.uncertainty) and self.uncertainty >= 0):
            raise ValueError(f"{self.name}: niepewność musi być skończona i >= 0")

    @property
    def params(self) -> Tuple[float, float, float, float]:
        return (self.a, self.b, self.c, self.d)

    def umf(self, x):
        """Upper Membership Function."""
        s = self.uncertainty
        return trapezoid(x, self.a, self.b, self.c, self.d, -s, +s)

    def lmf(self, x):
        """Lower Membership Function."""
        s = self.uncertainty
        return trapezoid(x, self.a, self.b, self.c, self.d, +s, -s)

    def interval(self, x):
        """Przedział przynależności [LMF(x), UMF(x)]."""
        return self.lmf(x), self.umf(x)

    def fou_width(self, x):
        """Szerokość FOU w punkcie x: UMF(x) - LMF(x) (zawsze >= 0)."""
        return self.umf(x) - self.lmf(x)

    def with_uncertainty(self, s: float) -> "IT2TrapezoidMF":
        return replace(self, uncertainty=float(s))
