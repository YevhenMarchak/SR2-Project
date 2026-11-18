import numpy as np
import pytest

from core.controller import COOLING, HEATING, NEUTRAL, IT2FuzzyController


@pytest.fixture(scope="module")
def ctrl():
    return IT2FuzzyController(1.0)


def test_heating_when_cold(ctrl):
    r = ctrl.compute(18, 22, 50)
    assert r.crisp < 0 and r.state == HEATING and r.temp_error == pytest.approx(-4)


def test_cooling_when_warm(ctrl):
    r = ctrl.compute(26, 22, 50)
    assert r.crisp > 0 and r.state == COOLING


def test_neutral_in_comfort_zone(ctrl):
    for h in (0, 30, 50):
        r = ctrl.compute(22, 22, h)
        assert abs(r.crisp) < 3 and r.state == NEUTRAL
    # przy wysokiej wilgotności reguła Comfort/Very High może lekko schłodzić, ale tylko delikatnie
    assert abs(ctrl.compute(22, 22, 70).crisp) < 10


def test_output_antisymmetry_at_optimal_humidity(ctrl):
    for d in (0.3, 1, 2, 3, 5):
        assert ctrl.compute(22 + d, 22, 50).crisp == pytest.approx(-ctrl.compute(22 - d, 22, 50).crisp, abs=1e-6)


def test_monotonic_in_temperature(ctrl):
    u = [ctrl.control(t, 22, 50) for t in np.linspace(14, 30, 161)]
    assert np.all(np.diff(u) >= -1e-9)


def test_humidity_modifies_output(ctrl):
    assert ctrl.compute(25, 22, 95).crisp > ctrl.compute(25, 22, 10).crisp   # wilgotno -> mocniej chłodzi
    assert ctrl.compute(20, 22, 95).crisp > ctrl.compute(20, 22, 10).crisp   # wilgotno -> słabiej grzeje (u mniej ujemne)


def test_humidity_alone_does_not_cause_strong_action(ctrl):
    assert abs(ctrl.compute(22, 22, 100).crisp) < 40
    assert abs(ctrl.compute(22, 22, 0).crisp) < 3


def test_output_range_and_finite(ctrl):
    for t in (-100, 0, 10, 22, 40, 500):
        for s in (16, 22, 30):
            for h in (-20, 0, 50, 100, 300):
                r = ctrl.compute(t, s, h)
                assert -100 <= r.crisp <= 100 and np.isfinite(r.crisp)
                assert r.y_left <= r.y_right + 1e-9


def test_diagnostics_complete(ctrl):
    r = ctrl.compute(20.3, 22, 62)
    assert set(r.temp_memberships) == set(ctrl.temperature.term_names)
    assert set(r.humidity_memberships) == set(ctrl.humidity.term_names)
    assert len(r.activations) == 20 and len(r.f_lower) == 20 == len(r.f_upper)
    assert r.type_reduction is not None and r.type_reduction.left.converged
    assert r.y_left <= r.crisp <= r.y_right
    assert r.crisp == pytest.approx((r.y_left + r.y_right) / 2)
    assert r.temp_error == pytest.approx(r.t_room - r.t_set)
    assert r.aggregated is not None and len(r.clipped) == len(r.active_rules)


def test_error_clipped_but_raw_kept(ctrl):
    r = ctrl.compute(40, 16, 50)
    assert r.temp_error_raw == 24 and r.temp_error == 10


def test_rejects_non_finite(ctrl):
    with pytest.raises(ValueError):
        ctrl.compute(float("nan"), 22, 50)


def test_fou_changes_interval_width_not_sign():
    narrow, wide = IT2FuzzyController(0.0), IT2FuzzyController(2.0)
    rn, rw = narrow.compute(21, 22, 60), wide.compute(21, 22, 60)
    assert (rw.y_right - rw.y_left) > (rn.y_right - rn.y_left) - 1e-12
    assert rn.y_right - rn.y_left == pytest.approx(0, abs=1e-6)   # typ 1 -> przedział zdegenerowany
    assert rn.crisp < 0 and rw.crisp < 0


def test_km_is_used(monkeypatch):
    import core.controller as mod
    calls = []
    orig = mod.type_reduce_center_of_sets
    monkeypatch.setattr(mod, "type_reduce_center_of_sets", lambda *a: calls.append(1) or orig(*a))
    IT2FuzzyController(1.0).compute(20, 22, 50)
    assert calls
