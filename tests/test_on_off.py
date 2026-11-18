import pytest

from classical.on_off_controller import COOL, HEAT, OFF, OnOffController
from core.room_simulation import RoomParams, simulate


def test_heating_hysteresis_cycle():
    c = OnOffController(hysteresis=0.5)
    assert c.step(22.0, 22) == 0 and c.state == OFF         # w pasie histerezy -- brak włączenia
    assert c.step(21.6, 22) == 0
    assert c.step(21.5, 22) == -100 and c.state == HEAT     # T <= Tset - h -> grzanie
    assert c.step(21.9, 22) == -100                         # nadal grzeje (pamięć stanu)
    assert c.step(22.0, 22) == 0 and c.state == OFF         # T >= Tset -> wyłączenie


def test_cooling_hysteresis_cycle():
    c = OnOffController(hysteresis=0.5)
    assert c.step(22.4, 22) == 0
    assert c.step(22.5, 22) == 100 and c.state == COOL
    assert c.step(22.1, 22) == 100
    assert c.step(22.0, 22) == 0 and c.state == OFF


def test_cooling_can_be_disabled():
    c = OnOffController(0.5, use_cooling=False)
    assert c.step(30, 22) == 0


def test_output_range_and_validation():
    c = OnOffController(1.0, power=60)
    assert {c.step(t, 22) for t in (10, 22, 35)} <= {-60.0, 0.0, 60.0}
    with pytest.raises(ValueError):
        OnOffController(0)
    with pytest.raises(ValueError):
        OnOffController(0.5, power=150)


def test_reset_clears_state():
    c = OnOffController(); c.step(10, 22); c.reset()
    assert c.state == OFF


def test_no_excessive_switching_in_simulation():
    d = simulate(OnOffController(0.5), RoomParams(duration_min=600)).data
    switches = int((d["u"].diff().abs() > 1).sum())
    assert switches < 60           # histereza ogranicza przełączanie
    wide = simulate(OnOffController(2.0), RoomParams(duration_min=600)).data
    assert int((wide["u"].diff().abs() > 1).sum()) <= switches   # szersza histereza -> nie więcej przełączeń


def test_regulates_temperature_around_setpoint():
    d = simulate(OnOffController(0.5), RoomParams(t0=17, t_set=22, t_out=12, duration_min=600)).data
    tail = d["t_room"].iloc[200:]
    assert tail.min() > 22 - 1.0 and tail.max() < 22 + 1.0
