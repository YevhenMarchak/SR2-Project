import numpy as np
import pytest

from core.controller import IT2FuzzyController
from core.room_simulation import RoomParams, heating_power, room_step, simulate


class Const:
    def __init__(self, u): self.u = u
    def control(self, t, s, h): return self.u


def run(u, **kw):
    return simulate(Const(u), RoomParams(**kw)).data


def test_heating_raises_temperature():
    d = run(-100, t0=20, t_out=20, duration_min=60)
    assert d["t_room"].iloc[-1] > 20 and np.all(np.diff(d["t_room"]) > 0)


def test_cooling_lowers_temperature():
    d = run(100, t0=25, t_out=25, duration_min=60)
    assert d["t_room"].iloc[-1] < 25 and np.all(np.diff(d["t_room"]) < 0)


def test_no_control_relaxes_to_outdoor_temperature():
    d = run(0, t0=25, t_out=10, duration_min=1440, tau_min=60)
    assert d["t_room"].iloc[-1] == pytest.approx(10, abs=1e-3)
    assert np.all(np.diff(d["t_room"]) < 0)


def test_outdoor_warmer_heats_room():
    assert run(0, t0=15, t_out=30, duration_min=60)["t_room"].iloc[-1] > 15


def test_steady_state_matches_theory():
    d = simulate(Const(-50), RoomParams(t_out=10, tau_min=100, heat_rate=0.2, duration_min=1440)).data
    assert d["t_room"].iloc[-1] == pytest.approx(20.0, abs=1e-3)


def test_exact_integration_independent_of_dt():
    a = run(-60, dt_min=0.5, duration_min=120)["t_room"].iloc[-1]
    b = run(-60, dt_min=5.0, duration_min=120)["t_room"].iloc[-1]
    assert a == pytest.approx(b, abs=1e-6)


def test_time_axis_and_length():
    p = RoomParams(duration_min=60, dt_min=2.0)
    d = simulate(Const(0), p).data
    assert len(d) == 31 and d["time_min"].iloc[0] == 0 and d["time_min"].iloc[-1] == 60
    assert np.allclose(np.diff(d["time_min"]), 2.0)
    assert np.all(np.isfinite(d.values.astype(float)))


def test_deterministic_by_default_and_seeded_noise():
    a, b = run(-30), run(-30)
    assert a.equals(b)
    n1 = run(0, noise_std=0.1, seed=1); n2 = run(0, noise_std=0.1, seed=1); n3 = run(0, noise_std=0.1, seed=2)
    assert n1.equals(n2) and not n1.equals(n3)


def test_parameter_limits_enforced():
    for bad in ({"dt_min": 0.0}, {"tau_min": 1.0}, {"heat_rate": -1}, {"humidity": 150},
                {"duration_min": 100000}, {"t_set": 5.0}, {"dt_min": float("nan")}):
        with pytest.raises(ValueError):
            RoomParams(**bad)
    assert RoomParams.clamped(dt_min=0.0, tau_min=1e9).dt_min == 0.1


def test_sign_convention_in_power():
    p = RoomParams()
    assert heating_power(-100, p) > 0 > heating_power(100, p) and heating_power(0, p) == 0
    assert heating_power(-500, p) == heating_power(-100, p)   # saturacja


def test_closed_loop_fuzzy_stabilises_without_oscillation():
    r = simulate(IT2FuzzyController(1.0), RoomParams(t0=17, t_set=22, t_out=12, duration_min=360))
    d = r.data
    assert abs(d["error"].iloc[-1]) < 1.0
    assert d["error"].iloc[-1] > -1.0 and (d["t_room"].diff().iloc[5:] >= -1e-6).all()
    assert d["u"].iloc[0] < 0 and d["heating"].iloc[0] > 0


def test_closed_loop_cooling_case():
    d = simulate(IT2FuzzyController(1.0), RoomParams(t0=28, t_set=22, t_out=30, duration_min=360)).data
    assert d["u"].iloc[0] > 0 and d["t_room"].iloc[-1] < 28
