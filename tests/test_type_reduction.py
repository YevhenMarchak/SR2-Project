import itertools

import numpy as np
import pytest

from core.type_reduction import (it2_centroid, km_endpoint, km_endpoints,
                                 type_reduce_center_of_sets)


def brute_force(y, fl, fu):
    """Referencja: przegląd wszystkich 2^N wyborów f_i in {f_l, F_l}."""
    best_min, best_max = np.inf, -np.inf
    for choice in itertools.product([0, 1], repeat=len(y)):
        f = np.where(np.array(choice) == 0, fl, fu)
        if f.sum() == 0:
            continue
        v = np.dot(f, y) / f.sum()
        best_min, best_max = min(best_min, v), max(best_max, v)
    return best_min, best_max


@pytest.mark.parametrize("seed", range(30))
def test_km_matches_bruteforce(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 9))
    y = rng.uniform(-100, 100, n)
    fu = rng.uniform(0.05, 1, n)
    fl = fu * rng.uniform(0, 1, n)
    left, right = km_endpoints(y, fl, fu)
    bmin, bmax = brute_force(y, fl, fu)
    assert left.value == pytest.approx(bmin, abs=1e-9)
    assert right.value == pytest.approx(bmax, abs=1e-9)
    assert left.converged and right.converged
    assert left.iterations <= n + 1 and right.iterations <= n + 1


def test_zero_width_intervals_give_weighted_average():
    y = np.array([-50.0, 0.0, 30.0]); f = np.array([0.2, 0.5, 0.9])
    l, r = km_endpoints(y, f, f)
    expected = np.dot(f, y) / f.sum()
    assert l.value == pytest.approx(expected) and r.value == pytest.approx(expected)


def test_left_le_right_and_inside_range():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = 6
        y = rng.uniform(-100, 100, n); fu = rng.uniform(0.01, 1, n); fl = fu * rng.uniform(0, 1, n)
        l, r = km_endpoints(y, fl, fu)
        assert l.value <= r.value + 1e-12
        assert y.min() - 1e-9 <= l.value and r.value <= y.max() + 1e-9


def test_zero_firing_strengths_are_ignored():
    y = np.array([-80.0, 10.0, 60.0]); fl = np.array([0.0, 0.0, 0.0]); fu = np.array([0.0, 0.4, 0.0])
    l, r = km_endpoints(y, fl, fu)
    assert l.value == pytest.approx(10.0) and r.value == pytest.approx(10.0)
    assert l.switch_point == 0  # pojedynczy aktywny element


def test_all_zero_raises():
    with pytest.raises(ValueError):
        km_endpoint([1.0, 2.0], [0, 0], [0, 0], "left")


def test_invalid_intervals_raise():
    with pytest.raises(ValueError):
        km_endpoint([1.0, 2.0], [0.9, 0.1], [0.5, 0.2], "left")  # f_l > F_l
    with pytest.raises(ValueError):
        km_endpoint([1.0, np.nan], [0.1, 0.1], [0.5, 0.2], "right")
    with pytest.raises(ValueError):
        km_endpoint([], [], [], "right")
    with pytest.raises(ValueError):
        km_endpoint([1.0], [0.1], [0.2], "middle")


def test_equal_y_values_and_single_rule():
    l, r = km_endpoints([5.0, 5.0, 5.0], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    assert l.value == pytest.approx(5.0) and r.value == pytest.approx(5.0)
    l, r = km_endpoints([7.0], [0.2], [0.9])
    assert l.value == r.value == 7.0


def test_unsorted_input_gives_same_result_as_sorted():
    rng = np.random.default_rng(3)
    y = rng.uniform(-100, 100, 7); fu = rng.uniform(0.1, 1, 7); fl = fu * 0.4
    perm = rng.permutation(7)
    a = km_endpoints(y, fl, fu); b = km_endpoints(y[perm], fl[perm], fu[perm])
    assert a[0].value == pytest.approx(b[0].value) and a[1].value == pytest.approx(b[1].value)


def test_center_of_sets_uses_left_and_right_centroids():
    c_l = np.array([-60.0, 0.0]); c_r = np.array([-50.0, 10.0])
    res = type_reduce_center_of_sets(c_l, c_r, [0.5, 0.5], [0.5, 0.5])
    assert res.y_left == pytest.approx(-30.0) and res.y_right == pytest.approx(-20.0)
    assert res.crisp == pytest.approx(-25.0)
    with pytest.raises(ValueError):
        type_reduce_center_of_sets(c_r, c_l, [0.5, 0.5], [0.5, 0.5])


def test_it2_centroid_symmetric_set():
    x = np.linspace(-1, 1, 401)
    umf = np.clip(1 - np.abs(x), 0, 1); lmf = 0.5 * umf
    cl, cr = it2_centroid(x, lmf, umf)
    assert cl == pytest.approx(-cr, abs=1e-6) and cl <= 0 <= cr
    cl1, cr1 = it2_centroid(x, umf, umf)   # typ 1: centroid = 0
    assert cl1 == pytest.approx(0, abs=1e-9) and cr1 == pytest.approx(0, abs=1e-9)
