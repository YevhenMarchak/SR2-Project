"""Testy interfejsu Streamlit (AppTest): przyciski, granice zakresów, aktualizacja wyniku."""
import re
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def make_app() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=120).run()


def output_value(at: AppTest) -> float:
    for md in at.markdown:
        m = re.search(r'Controller output u</div><div class="big">([+-]?[0-9.]+)', md.value)
        if m:
            return float(m.group(1))
    raise AssertionError("Nie znaleziono karty wyjścia regulatora")


def state_badge(at: AppTest) -> str:
    for md in at.markdown:
        m = re.search(r'class="badge (HEATING|COOLING|NEUTRAL)"', md.value)
        if m:
            return m.group(1)
    raise AssertionError("Brak statusu")


def click(at: AppTest, key: str) -> AppTest:
    next(b for b in at.button if b.key == key).click()
    return at.run()


def test_app_starts_without_exceptions():
    at = make_app()
    assert not at.exception
    assert len(at.tabs) == 7


def test_temperature_buttons_update_state_and_output():
    at = make_app()
    assert at.session_state["T_room"] == 19.0 and state_badge(at) == "HEATING"
    u0 = output_value(at)
    at = click(at, "T_room_inc")
    assert at.session_state["T_room"] == 19.5
    assert output_value(at) > u0            # cieplej -> mniej grzania
    for _ in range(5):
        at = click(at, "T_room_inc")
    assert at.session_state["T_room"] == 22.0 and state_badge(at) == "NEUTRAL"
    for _ in range(8):
        at = click(at, "T_room_inc")
    assert state_badge(at) == "COOLING" and output_value(at) > 0
    at = click(at, "T_room_dec")
    assert at.session_state["T_room"] == 25.5


def test_humidity_buttons_change_output():
    at = make_app()
    at.session_state["T_room"] = 25.0
    at = at.run()
    u_before = output_value(at)
    for _ in range(40):
        at = click(at, "humidity_inc")
    assert at.session_state["humidity"] == 95.0
    assert output_value(at) > u_before      # wilgotno -> mocniejsze chłodzenie


def test_setpoint_buttons_change_error_and_output():
    at = make_app()
    at.session_state["T_room"] = 22.0
    at = at.run()
    assert state_badge(at) == "NEUTRAL"
    for _ in range(4):
        at = click(at, "T_set_dec")         # T_set 20 -> pokój cieplejszy niż zadana
    assert at.session_state["T_set"] == 20.0 and state_badge(at) == "COOLING"


def test_range_limits_respected():
    at = make_app()
    at.session_state["T_room"], at.session_state["humidity"], at.session_state["T_set"] = 40.0, 100.0, 30.0
    at = at.run()
    for key in ("T_room_inc", "humidity_inc", "T_set_inc"):
        at = click(at, key)
    assert (at.session_state["T_room"], at.session_state["humidity"], at.session_state["T_set"]) == (40.0, 100.0, 30.0)
    at.session_state["T_room"], at.session_state["humidity"], at.session_state["T_set"] = 10.0, 0.0, 16.0
    at = at.run()
    for key in ("T_room_dec", "humidity_dec", "T_set_dec"):
        at = click(at, key)
    assert (at.session_state["T_room"], at.session_state["humidity"], at.session_state["T_set"]) == (10.0, 0.0, 16.0)
    assert not at.exception


def test_fou_slider_updates_app():
    at = make_app()
    at.session_state["fou_factor"] = 2.0
    at = at.run()
    assert not at.exception
    assert any("FOU ×2" in md.value for md in at.markdown)


def test_simulation_parameters_validated_without_crash():
    at = make_app()
    at.session_state["sim_t0"] = 15.0
    at.session_state["sim_duration"] = 120.0
    at = at.run()
    assert not at.exception


def test_all_tabs_render_without_exceptions():
    """Tryb nieleniwy: renderuje wszystkie 7 zakładek naraz."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_lazy"] = False
    at.run()
    assert not at.exception
    text = " ".join([md.value for md in at.markdown] + [str(h.value) for h in at.subheader]
                    + [str(c.value) for c in at.caption])
    for needle in ("Fuzzification", "Active rules", "Impact of uncertainty", "Karnik"):
        assert needle in text
    # tabele metryk (symulacja + porównanie) zawierają wymagane wskaźniki
    tables = [df.value for df in at.dataframe]
    assert any("MAE [°C]" in t.index and "Energy proxy [full-power min]" in t.index for t in tables)
    assert any(set(t.columns) == {"IT2FLS", "ON/OFF"} for t in tables)
    # pełna baza reguł (20 wierszy) jest w tabeli
    assert any(len(t) == 20 and "ID" in t.columns for t in tables)


def test_settings_persist_across_lazy_runs():
    at = make_app()
    at.session_state["fou_factor"] = 0.5
    at.session_state["sim_t0"] = 14.0
    at = at.run()
    at = click(at, "T_room_inc")
    assert at.session_state["fou_factor"] == 0.5 and at.session_state["sim_t0"] == 14.0
