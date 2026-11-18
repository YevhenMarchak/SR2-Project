"""Klasyczny regulator ON/OFF z histerezą (grzanie i chłodzenie).

Konwencja wyjścia identyczna jak w IT2FLS: u = -power (ogrzewanie),
u = +power (chłodzenie), u = 0 (wyłączony).

Logika (h = histereza, T_set = temperatura zadana):
    OFF  -> HEAT gdy T <= T_set - h;     HEAT -> OFF gdy T >= T_set
    OFF  -> COOL gdy T >= T_set + h;     COOL -> OFF gdy T <= T_set
Stan jest zapamiętany między wywołaniami, dzięki czemu regulator nie przełącza
się przy każdym kroku w pobliżu progu (to właśnie istota histerezy).
"""
from __future__ import annotations

OFF, HEAT, COOL = "OFF", "HEAT", "COOL"


class OnOffController:
    def __init__(self, hysteresis: float = 0.5, power: float = 100.0, use_cooling: bool = True) -> None:
        if not hysteresis > 0:
            raise ValueError("Histereza musi być > 0")
        if not 0 < power <= 100:
            raise ValueError("Moc musi należeć do (0, 100]")
        self.hysteresis = float(hysteresis)
        self.power = float(power)
        self.use_cooling = use_cooling
        self.state = OFF

    def reset(self) -> None:
        self.state = OFF

    def step(self, t_room: float, t_set: float) -> float:
        h = self.hysteresis
        if self.state == HEAT and t_room >= t_set:
            self.state = OFF
        elif self.state == COOL and t_room <= t_set:
            self.state = OFF
        if self.state == OFF:
            if t_room <= t_set - h:
                self.state = HEAT
            elif self.use_cooling and t_room >= t_set + h:
                self.state = COOL
        return {HEAT: -self.power, COOL: self.power, OFF: 0.0}[self.state]

    def control(self, t_room: float, t_set: float, humidity: float = 0.0) -> float:
        """Interfejs zgodny z regulatorem IT2FLS (wilgotność jest ignorowana)."""
        return self.step(t_room, t_set)
