import numpy as np
import pytest

from classical.on_off_controller import OnOffController
from core.analysis import control_surface, fou_impact, response_curve
from core.controller import IT2FuzzyController
from core.room_simulation import RoomParams, simulate
from visualization import comparison_plots as cp
from visualization import inference_plots as ip
from visualization import membership_plots as mp
from visualization import simulation_plots as sp


@pytest.fixture(scope="module")
def ctrl():
    return IT2FuzzyController(1.0)


def test_response_curve_comes_from_engine(ctrl):
    e = np.linspace(-6, 6, 13)
    c = response_curve(ctrl, e, 55.0, t_set=22.0)
    for i in (0, 3, 6, 9, 12):
        assert c["crisp"][i] == pytest.approx(ctrl.compute(22 + e[i], 22, 55).crisp)
    assert np.all(c["y_left"] <= c["crisp"] + 1e-9) and np.all(c["crisp"] <= c["y_right"] + 1e-9)


def test_control_surface_shape_and_values(ctrl):
    e, h = np.linspace(-4, 4, 5), np.linspace(0, 100, 3)
    z = control_surface(ctrl, e, h)
    assert z.shape == (3, 5)
    assert z[1, 3] == pytest.approx(ctrl.compute(22 + e[3], 22, h[1]).crisp)
    assert np.all(np.abs(z) <= 100)


def test_fou_impact_widens_interval():
    curves = fou_impact([0.0, 1.0, 2.0], np.linspace(-3, 3, 25), 50)
    w = [float(np.sum(curves[f]["y_right"] - curves[f]["y_left"])) for f in (0.0, 1.0, 2.0)]
    assert w[0] < 1e-6 < w[1] < w[2]


def test_all_plots_render(ctrl):
    import matplotlib.pyplot as plt
    r = ctrl.compute(20.5, 22, 60)
    e, h = np.linspace(-6, 6, 9), np.linspace(0, 100, 5)
    p = RoomParams(duration_min=60)
    a, b = simulate(ctrl, p).data, simulate(OnOffController(), p).data
    figs = [mp.plot_temperature(ctrl.temperature, -1.5), mp.plot_humidity(ctrl.humidity, 60),
            mp.plot_output(ctrl.output, r.crisp), mp.plot_output(ctrl.output),
            ip.plot_response_curve(response_curve(ctrl, e, 60), -1.5, 60),
            ip.plot_heatmap(e, h, control_surface(ctrl, e, h), (-1.5, 60)),
            ip.plot_firing_strengths(r), ip.plot_aggregation(r),
            sp.plot_temperature(a), sp.plot_control(a),
            cp.plot_temperature_comparison({"IT2FLS": a, "ON/OFF": b}),
            cp.plot_control_comparison({"IT2FLS": a, "ON/OFF": b}),
            cp.plot_fou_impact(fou_impact([0, 1], e, 50))]
    assert all(f.axes for f in figs)
    for f in figs:
        plt.close(f)
