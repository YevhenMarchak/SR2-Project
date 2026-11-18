import numpy as np
import pytest

from config import default_config as cfg
from core.membership_type2 import IT2TrapezoidMF
from core.variables import build_variables


@pytest.mark.parametrize("factor", [0.0, 0.5, 1.0, 2.0, 3.0])
def test_lmf_le_umf_and_range(factor):
    for var in build_variables(factor):
        x = np.linspace(*var.universe, 2001)
        lo, hi = var.memberships(x)
        assert np.all(lo <= hi + 1e-12)
        assert np.all((lo >= 0) & (hi <= 1))
        assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))


def test_zero_uncertainty_gives_type1():
    for var in build_variables(0.0):
        x = np.linspace(*var.universe, 501)
        lo, hi = var.memberships(x)
        assert np.allclose(lo, hi)


def test_fou_is_umf_minus_lmf_and_positive():
    mf = IT2TrapezoidMF("t", -2, 0, 0, 2, uncertainty=0.3)
    x = np.linspace(-3, 3, 601)
    assert np.allclose(mf.fou_width(x), mf.umf(x) - mf.lmf(x))
    assert mf.fou_width(0.0) > 0            # FOU także w szczycie trójkąta
    assert mf.umf(0.0) == pytest.approx(1.0)
    assert mf.lmf(0.0) == pytest.approx(1 - 0.3 / 2)


def test_fou_grows_with_uncertainty():
    x = np.linspace(-10, 10, 1001)
    areas = []
    for f in (0.0, 1.0, 2.0):
        t = build_variables(f)[0]
        lo, hi = t.memberships(x)
        areas.append(float(np.sum(hi - lo)))
    assert areas[0] == 0 and areas[0] < areas[1] < areas[2]


def test_boundary_values_of_shapes():
    mf = IT2TrapezoidMF("trap", 0, 1, 2, 3)
    assert mf.umf(-0.5) == 0 and mf.umf(1.5) == 1 and mf.umf(3.5) == 0
    assert float(mf.umf(0.5)) == pytest.approx(0.5)
    sh = IT2TrapezoidMF("shoulder", -np.inf, -np.inf, -1, 1)
    assert sh.umf(-1e6) == 1 and sh.lmf(-1e6) == 1


def test_coverage_every_input_has_membership():
    """Brak luk: dla każdego x suma/maksimum UMF > 0 w każdej zmiennej."""
    for var in build_variables(1.0)[:2]:
        x = np.linspace(*var.universe, 2001)
        _, hi = var.memberships(x)
        assert np.all(hi.max(axis=0) > 0.4)


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        IT2TrapezoidMF("bad", 2, 1, 3, 4)
    with pytest.raises(ValueError):
        IT2TrapezoidMF("bad", 0, 1, 2, 3, uncertainty=-0.1)
    with pytest.raises(ValueError):
        IT2TrapezoidMF("bad", 0, np.nan, 2, 3)
    with pytest.raises(ValueError):
        build_variables(-1.0)


def test_inputs_outside_universe_are_clipped():
    t = build_variables(1.0)[0]
    lo1, hi1 = t.memberships(-1e6)
    lo2, hi2 = t.memberships(t.universe[0])
    assert np.allclose(lo1, lo2) and np.allclose(hi1, hi2)
