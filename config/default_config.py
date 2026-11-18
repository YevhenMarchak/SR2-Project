"""Centralna konfiguracja Fuzzy HVAC Simulator.

Konwencje (stosowane w całym projekcie):
    e = T_room - T_set      (e < 0 -> za zimno, e > 0 -> za ciepło)
    u in [-100, +100]       (u < 0 -> ogrzewanie, u > 0 -> chłodzenie)

Każdy zbiór rozmyty to trapez (a, b, c, d); ``-inf``/``+inf`` w parametrach
oznaczają zbiór "ramieniowy" (shoulder), który sięga do krawędzi dziedziny.
Trójkąt to trapez z b == c.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

INF = float("inf")

# ----------------------------------------------------------------------------
# Zakresy
# ----------------------------------------------------------------------------
TEMP_ERROR_RANGE: Tuple[float, float] = (-10.0, 10.0)   # [°C]
HUMIDITY_RANGE: Tuple[float, float] = (0.0, 100.0)      # [%RH]
OUTPUT_RANGE: Tuple[float, float] = (-100.0, 100.0)     # [-] (jednostki sterowania)
SETPOINT_RANGE: Tuple[float, float] = (16.0, 30.0)      # [°C]
ROOM_TEMP_RANGE: Tuple[float, float] = (10.0, 40.0)     # [°C]

TEMP_STEP = 0.5
HUMIDITY_STEP = 1.0

# ----------------------------------------------------------------------------
# Funkcje przynależności (nominalne trapezy, zanim doliczymy FOU)
# ----------------------------------------------------------------------------
TEMPERATURE_TERMS: Dict[str, Tuple[float, float, float, float]] = {
    "Very Cold": (-INF, -INF, -4.0, -2.0),
    "Cold":      (-4.0, -2.0, -2.0, -0.4),
    "Comfort":   (-2.0, -0.4, 0.4, 2.0),
    "Warm":      (0.4, 2.0, 2.0, 4.0),
    "Very Warm": (2.0, 4.0, INF, INF),
}

HUMIDITY_TERMS: Dict[str, Tuple[float, float, float, float]] = {
    "Low":       (-INF, -INF, 25.0, 40.0),
    "Optimal":   (25.0, 40.0, 55.0, 70.0),
    "High":      (55.0, 70.0, 70.0, 85.0),
    "Very High": (70.0, 85.0, INF, INF),
}

OUTPUT_TERMS: Dict[str, Tuple[float, float, float, float]] = {
    "Strong Heating":   (-INF, -INF, -90.0, -65.0),
    "Moderate Heating": (-90.0, -65.0, -65.0, -30.0),
    "Gentle Heating":   (-65.0, -30.0, -30.0, 0.0),
    "Neutral":          (-30.0, 0.0, 0.0, 30.0),
    "Gentle Cooling":   (0.0, 30.0, 30.0, 65.0),
    "Moderate Cooling": (30.0, 65.0, 65.0, 90.0),
    "Strong Cooling":   (65.0, 90.0, INF, INF),
}

# ----------------------------------------------------------------------------
# Footprint of Uncertainty
# ----------------------------------------------------------------------------
# Przesunięcie krawędzi UMF/LMF (w jednostkach dziedziny) dla współczynnika = 1.
FOU_SCALE: Dict[str, float] = {"temperature": 0.25, "humidity": 3.0, "output": 4.0}
DEFAULT_FOU_FACTOR = 1.0
FOU_FACTOR_RANGE: Tuple[float, float] = (0.0, 3.0)

# ----------------------------------------------------------------------------
# Baza reguł: wiersze = temperatura, kolumny = wilgotność
# ----------------------------------------------------------------------------
RULE_TEMP_ORDER: List[str] = list(TEMPERATURE_TERMS)
RULE_HUM_ORDER: List[str] = list(HUMIDITY_TERMS)

RULE_TABLE: Dict[str, List[str]] = {
    #                 Low                 Optimal             High                Very High
    "Very Cold": ["Strong Heating",   "Strong Heating",   "Strong Heating",   "Moderate Heating"],
    "Cold":      ["Moderate Heating", "Moderate Heating", "Gentle Heating",   "Gentle Heating"],
    "Comfort":   ["Neutral",          "Neutral",          "Neutral",          "Gentle Cooling"],
    "Warm":      ["Gentle Cooling",   "Moderate Cooling", "Moderate Cooling", "Strong Cooling"],
    "Very Warm": ["Moderate Cooling", "Strong Cooling",   "Strong Cooling",   "Strong Cooling"],
}

# |u| poniżej progu => status NEUTRAL
NEUTRAL_THRESHOLD = 3.0
# Rozdzielczość dyskretyzacji dziedziny wyjścia (centroidy zbiorów konsekwencji, agregacja)
OUTPUT_GRID_POINTS = 2001

# ----------------------------------------------------------------------------
# Symulacja pomieszczenia (model demonstracyjny) -- czas w minutach
# ----------------------------------------------------------------------------
SIM_DEFAULTS = {
    "t_out": 12.0,           # temperatura zewnętrzna [°C]
    "t0": 17.0,              # temperatura początkowa [°C]
    "t_set": 22.0,           # temperatura zadana [°C]
    "humidity": 50.0,        # wilgotność (stała w symulacji) [%]
    "duration_min": 240.0,   # czas symulacji [min]
    "dt_min": 1.0,           # krok czasowy [min]
    "tau_min": 300.0,        # stała czasowa strat ciepła [min]
    "heat_rate": 0.25,       # maks. szybkość ogrzewania [°C/min] przy u = -100
    "cool_rate": 0.25,       # maks. szybkość chłodzenia [°C/min] przy u = +100
    "noise_std": 0.0,        # odchylenie std zakłóceń [°C/krok] (0 = deterministycznie)
    "seed": 42,
    "hysteresis": 0.5,       # histereza ON/OFF [°C]
    "comfort_band": 1.0,     # strefa komfortu +-[°C] (metryki)
}

SIM_LIMITS = {
    "t_out": (-20.0, 45.0),
    "t0": (10.0, 40.0),
    "t_set": SETPOINT_RANGE,
    "humidity": HUMIDITY_RANGE,
    "duration_min": (10.0, 1440.0),
    "dt_min": (0.1, 5.0),
    "tau_min": (20.0, 1000.0),
    "heat_rate": (0.01, 1.0),
    "cool_rate": (0.01, 1.0),
    "noise_std": (0.0, 0.5),
    "hysteresis": (0.1, 3.0),
    "comfort_band": (0.2, 3.0),
}
MAX_SIM_STEPS = 20000
