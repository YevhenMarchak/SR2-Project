import numpy as np
import pandas as pd
import pytest

from core import metrics as m


def test_mae_rmse_known_values():
    e = [3, -4]
    assert m.mae(e) == pytest.approx(3.5)
    assert m.rmse(e) == pytest.approx(np.sqrt(12.5))
    assert m.rmse([0, 0, 0]) == 0 and m.mae([2, 2]) == 2
    assert m.rmse([1, -2, 3]) >= m.mae([1, -2, 3])


def test_time_to_comfort_and_settling():
    t = np.arange(10.0); e = np.array([-5, -4, -3, -0.5, 0.2, 2.0, 0.3, 0.1, 0.0, 0.1])
    assert m.time_to_comfort(t, e, 1.0) == 3
    assert m.settling_time(t, e, 1.0) == 6          # po e=2.0 (t=5) już w paśmie
    assert m.time_to_comfort(t, np.full(10, 5.0), 1.0) is None
    assert m.settling_time(t, np.full(10, 5.0), 1.0) is None
    assert m.settling_time(t, np.zeros(10), 1.0) == 0


def test_overshoot_undershoot_heating():
    t = np.array([18, 20, 22.5, 23.0, 21.5, 22.0])
    over, under = m.overshoot_undershoot(t, 22)
    assert over == pytest.approx(1.0) and under == pytest.approx(0.5)


def test_overshoot_undershoot_cooling_and_none():
    t = np.array([26, 24, 21.4, 22.0])
    over, under = m.overshoot_undershoot(t, 22)
    assert over == pytest.approx(0.6) and under == 0
    assert m.overshoot_undershoot(np.array([18, 19, 20, 21.5]), 22) == (0.0, 0.0)


def test_control_changes_and_energy():
    u = np.array([0, 0, -100, -100, 0, 0, 100])
    assert m.count_control_changes(u, 5) == 3
    assert m.count_control_changes(np.zeros(5), 5) == 0
    assert m.energy_proxy([-100, -50, 50, 100], dt_min=2.0) == pytest.approx(6.0)
    assert m.energy_proxy([0, 0, 0], 1.0) == 0


def test_compute_metrics_keys_and_consistency():
    df = pd.DataFrame({"time_min": np.arange(5.0), "t_room": [20, 21, 22, 22.5, 22],
                       "t_set": 22.0, "u": [-80, -60, -10, 0, 0]})
    df["error"] = df["t_room"] - df["t_set"]
    out = m.compute_metrics(df, dt_min=1.0, comfort_band=1.0)
    assert out["MAE [°C]"] == pytest.approx(np.mean(np.abs(df["error"])))
    assert out["Time to comfort [min]"] == 1 and out["Settling time [min]"] == 1
    assert out["Overshoot [°C]"] == pytest.approx(0.5)
    assert out["Energy proxy [full-power min]"] == pytest.approx(1.5)
    assert len(out) == 8
