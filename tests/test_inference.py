import numpy as np
import pytest

from core.controller import IT2FuzzyController


@pytest.fixture(scope="module")
def ctrl():
    return IT2FuzzyController(1.0)


def test_firing_strength_intervals_valid(ctrl):
    for e in np.linspace(-12, 12, 49):
        for h in (0, 20, 50, 80, 100):
            lo, hi = ctrl.engine.fire(e, h)
            assert np.all(lo <= hi + 1e-12) and np.all(lo >= 0) and np.all(hi <= 1)
            assert hi.max() > 0   # zawsze odpala co najmniej jedna reguła


def test_min_tnorm_firing(ctrl):
    e, h = -1.0, 45.0
    lo, hi = ctrl.engine.fire(e, h)
    t_lo, t_hi = ctrl.temperature.memberships(e); h_lo, h_hi = ctrl.humidity.memberships(h)
    for i, r in enumerate(ctrl.engine.rules):
        ti, hi_ = ctrl.temperature.term_names.index(r.temperature_term), ctrl.humidity.term_names.index(r.humidity_term)
        assert lo[i] == pytest.approx(min(t_lo[ti], h_lo[hi_]))
        assert hi[i] == pytest.approx(min(t_hi[ti], h_hi[hi_]))


def test_only_expected_rules_active_at_exact_comfort(ctrl):
    res = ctrl.compute(22, 22, 50)
    active = {(a.rule.temperature_term, a.rule.humidity_term) for a in res.active_rules}
    assert ("Comfort", "Optimal") in active
    assert all(t != "Very Cold" and t != "Very Warm" for t, _ in active)


def test_extreme_inputs_activate_extreme_rules(ctrl):
    res = ctrl.compute(40, 16, 100)
    top = res.active_rules[0]
    assert top.rule.temperature_term == "Very Warm" and top.rule.humidity_term == "Very High"
    assert top.f_upper == pytest.approx(1.0)
    res = ctrl.compute(10, 30, 0)
    assert res.active_rules[0].rule.temperature_term == "Very Cold"


def test_aggregated_set_is_valid_interval_set(ctrl):
    res = ctrl.compute(20.5, 22, 60)
    x, lo, hi = res.aggregated
    assert np.all(lo <= hi + 1e-12) and hi.max() <= 1 and hi.max() > 0
    assert len(x) == len(lo) == len(hi)


def test_centroids_ordered_and_symmetric(ctrl):
    cl, cr = ctrl.engine.centroid_left, ctrl.engine.centroid_right
    assert np.all(cl <= cr + 1e-9)
    assert np.all(np.diff(cl) > 0)
    assert np.allclose(cl, -cr[::-1], atol=1e-6)   # symetria wyjścia
