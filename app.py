"""Fuzzy HVAC Simulator -- panel Streamlit dla regulatora Interval Type-2 FLS.

Cała logika (inferencja, KM, symulacja, metryki) żyje w pakietach ``core``,
``classical`` i ``visualization``; ten plik tylko wywołuje silnik i prezentuje wyniki.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from classical.on_off_controller import OnOffController
from config import default_config as cfg
from core.analysis import control_surface, fou_impact, response_curve
from core.controller import COOLING, HEATING, NEUTRAL, IT2FuzzyController
from core.metrics import compute_metrics
from core.room_simulation import RoomParams, simulate
from core.rule_base import RULES, rules_to_dataframe
from visualization import comparison_plots as cmp_plots
from visualization import inference_plots as inf_plots
from visualization import membership_plots as mf_plots
from visualization import simulation_plots as sim_plots

st.set_page_config(page_title="Fuzzy HVAC Simulator", page_icon="🌡️", layout="wide")

# ----------------------------------------------------------------------------
# Stan sesji
# ----------------------------------------------------------------------------
DEFAULTS = {
    "T_room": 19.0, "humidity": 55.0, "T_set": cfg.SIM_DEFAULTS["t_set"],
    "fou_factor": cfg.DEFAULT_FOU_FACTOR, "hysteresis": cfg.SIM_DEFAULTS["hysteresis"],
    "comfort_band": cfg.SIM_DEFAULTS["comfort_band"],
    "sim_t0": cfg.SIM_DEFAULTS["t0"], "sim_t_set": cfg.SIM_DEFAULTS["t_set"],
    "sim_t_out": cfg.SIM_DEFAULTS["t_out"], "sim_humidity": cfg.SIM_DEFAULTS["humidity"],
    "sim_duration": cfg.SIM_DEFAULTS["duration_min"], "sim_dt": cfg.SIM_DEFAULTS["dt_min"],
    "sim_tau": cfg.SIM_DEFAULTS["tau_min"], "sim_heat": cfg.SIM_DEFAULTS["heat_rate"],
    "sim_cool": cfg.SIM_DEFAULTS["cool_rate"], "sim_noise": cfg.SIM_DEFAULTS["noise_std"],
}
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)
    st.session_state[_k] = st.session_state[_k]  # zachowaj wartości widgetów z nieaktywnych zakładek


def adjust(key: str, delta: float, lo: float, hi: float) -> None:
    """Callback przycisków +/-: zmienia wartość i pilnuje zakresu."""
    st.session_state[key] = round(float(np.clip(st.session_state[key] + delta, lo, hi)), 2)


def reset_settings() -> None:
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


# ----------------------------------------------------------------------------
# Cache obliczeń (wszystko liczy prawdziwy silnik)
# ----------------------------------------------------------------------------
@st.cache_resource
def get_controller(fou: float) -> IT2FuzzyController:
    return IT2FuzzyController(fou)


@st.cache_data(show_spinner=False)
def cached_curve(fou: float, humidity: float):
    return response_curve(get_controller(fou), np.linspace(-10, 10, 81), humidity)


@st.cache_data(show_spinner=False)
def cached_surface(fou: float):
    e, h = np.linspace(-10, 10, 41), np.linspace(0, 100, 21)
    return e, h, control_surface(get_controller(fou), e, h)


@st.cache_data(show_spinner=False)
def cached_fou_impact(humidity: float):
    return fou_impact([0.0, 0.5, 1.0, 2.0, 3.0], np.linspace(-6, 6, 49), humidity)


@st.cache_data(show_spinner="Running simulation…")
def cached_runs(params: dict, fou: float, hysteresis: float):
    p = RoomParams(**params)
    fuzzy = simulate(get_controller(fou), p, "IT2FLS").data
    onoff = simulate(OnOffController(hysteresis), p, "ON/OFF").data
    return fuzzy, onoff


def show(fig) -> None:
    st.pyplot(fig, width="stretch")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Styl
# ----------------------------------------------------------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1280px;}
.hero {display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:.6rem;}
.hero h1 {margin:0; font-size:2rem;}
.pill {display:inline-block; padding:.18rem .7rem; border-radius:999px; font-size:.78rem; font-weight:600; margin-left:.35rem;}
.pill.ok {background:#12a59422; color:#0c7d70;}
.pill.sim {background:#f2c14e33; color:#8a6a00;}
.big {font-size:2.6rem; font-weight:700; line-height:1.1;}
.big small {font-size:1.1rem; font-weight:500; opacity:.65; margin-left:.2rem;}
.lbl {font-size:.82rem; opacity:.65; text-transform:uppercase; letter-spacing:.05em;}
.badge {display:inline-block; padding:.3rem .9rem; border-radius:10px; font-weight:700; font-size:1rem;}
.badge.HEATING {background:#e4572e22; color:#c23f17;}
.badge.COOLING {background:#2a6fdb22; color:#1b53b0;}
.badge.NEUTRAL {background:#8a8f9822; color:#5b616b;}
.gauge {position:relative; height:12px; border-radius:6px; margin:.9rem 0 .2rem;
        background:linear-gradient(90deg,#e4572e,#e9e9e9 50%,#2a6fdb);}
.gauge .mark {position:absolute; top:-5px; width:6px; height:22px; border-radius:3px; background:#111; transform:translateX(-3px);}
.gauge-lbl {display:flex; justify-content:space-between; font-size:.72rem; opacity:.65;}
</style>
""", unsafe_allow_html=True)

fou = float(st.session_state.fou_factor)
ctrl = get_controller(fou)
res = ctrl.compute(st.session_state.T_room, st.session_state.T_set, st.session_state.humidity)

ICON = {HEATING: "🔥", NEUTRAL: "⚖️", COOLING: "❄️"}


def describe(r) -> str:
    if r.state == NEUTRAL:
        return "Neutral"
    return f"{'Heating' if r.state == HEATING else 'Cooling'} {abs(r.crisp):.0f}%"


# ----------------------------------------------------------------------------
# Nagłówek
# ----------------------------------------------------------------------------
st.markdown(f"""
<div class="hero"><div><h1>🌡️ Fuzzy HVAC Simulator</h1>
<span style="opacity:.7">Interval Type-2 fuzzy temperature control – Mamdani inference + Karnik–Mendel type reduction</span></div>
<div><span class="pill ok">● System online</span><span class="pill sim">SIMULATOR – no physical device</span>
<span class="pill ok">FOU ×{fou:g}</span></div></div>
""", unsafe_allow_html=True)
st.write("")

def current_params() -> dict:
    s = st.session_state
    return dict(t_out=s.sim_t_out, t0=s.sim_t0, t_set=s.sim_t_set, humidity=s.sim_humidity,
                duration_min=s.sim_duration, dt_min=s.sim_dt, tau_min=s.sim_tau, heat_rate=s.sim_heat,
                cool_rate=s.sim_cool, noise_std=s.sim_noise, seed=cfg.SIM_DEFAULTS["seed"])


def fmt_metrics(d: dict) -> dict:
    return {k: ("—" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))) for k, v in d.items()}


def get_sim():
    """(params, fuzzy_df, onoff_df, error) -- liczone przez silnik, z cache."""
    params = current_params()
    try:
        RoomParams(**params)  # walidacja zakresów
        fuzzy_df, onoff_df = cached_runs(params, fou, float(st.session_state.hysteresis))
        return params, fuzzy_df, onoff_df, None
    except ValueError as exc:
        return params, None, None, str(exc)


TAB_NAMES = ["📊 Dashboard", "📈 Membership Functions", "📋 Rule Base", "🧠 Inference",
             "⏱️ Simulation", "⚖️ Comparison", "⚙️ Settings"]
# Tryb leniwy: wykonujemy kod tylko aktywnej zakładki (szybsza reakcja na kliknięcia).
LAZY = st.session_state.get("_lazy", True)
try:
    tabs = st.tabs(TAB_NAMES, on_change="rerun", key="main_tabs") if LAZY else st.tabs(TAB_NAMES)
except TypeError:  # starsza wersja Streamlit bez trybu leniwego
    tabs = st.tabs(TAB_NAMES)


def is_open(tab) -> bool:
    return (not LAZY) or getattr(tab, "open", True) is not False


# ============================================================================
# TAB 1: Dashboard
# ============================================================================
def tab_dashboard():
    def stepper(col, title, key, step, lo, hi, unit, fmt):
        with col, st.container(border=True):
            st.markdown(f'<div class="lbl">{title}</div><div class="big">{st.session_state[key]:{fmt}}<small>{unit}</small></div>',
                        unsafe_allow_html=True)
            b1, b2 = st.columns(2)
            b1.button("−", key=f"{key}_dec", on_click=adjust, args=(key, -step, lo, hi), width="stretch")
            b2.button("+", key=f"{key}_inc", on_click=adjust, args=(key, +step, lo, hi), width="stretch")

    c1, c2, c3 = st.columns(3)
    stepper(c1, "Room temperature", "T_room", cfg.TEMP_STEP, *cfg.ROOM_TEMP_RANGE, "°C", ".1f")
    stepper(c2, "Humidity", "humidity", cfg.HUMIDITY_STEP, *cfg.HUMIDITY_RANGE, "%", ".0f")
    stepper(c3, "Set temperature", "T_set", cfg.TEMP_STEP, *cfg.SETPOINT_RANGE, "°C", ".1f")

    o1, o2, o3 = st.columns([2, 1, 1])
    with o1, st.container(border=True):
        st.markdown(f'<div class="lbl">Controller output u</div><div class="big">{res.crisp:+.1f}</div>'
                    f'<span class="badge {res.state}">{ICON[res.state]} {res.state} · {describe(res)}</span>'
                    f'<div class="gauge"><div class="mark" style="left:{(res.crisp + 100) / 2:.1f}%"></div></div>'
                    f'<div class="gauge-lbl"><span>−100 max heating</span><span>0 neutral</span><span>+100 max cooling</span></div>',
                    unsafe_allow_html=True)
    with o2, st.container(border=True):
        st.metric("Temperature error e = T − T_set", f"{res.temp_error_raw:+.1f} °C")
        st.caption("e < 0: too cold → heating · e > 0: too warm → cooling")
    with o3, st.container(border=True):
        st.metric("Type-reduced interval", f"[{res.y_left:.1f}, {res.y_right:.1f}]")
        st.caption("crisp u = (y_left + y_right) / 2")

    g1, g2 = st.columns(2)
    with g1:
        show(inf_plots.plot_response_curve(cached_curve(fou, float(st.session_state.humidity)),
                                           res.temp_error, st.session_state.humidity))
    with g2:
        e_g, h_g, z = cached_surface(fou)
        show(inf_plots.plot_heatmap(e_g, h_g, z, (res.temp_error, res.humidity)))

    st.subheader("Active rules")
    rows = [{"Rule": f"R{a.rule.rule_id}", "IF temperature error": a.rule.temperature_term,
             "AND humidity": a.rule.humidity_term, "THEN output": a.rule.output_term,
             "f lower": round(a.f_lower, 3), "f upper": round(a.f_upper, 3)} for a in res.active_rules]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# ============================================================================
# TAB 2: Membership functions
# ============================================================================
def tab_membership():
    st.caption("Solid line = UMF, dashed = LMF, shaded area = Footprint of Uncertainty (FOU). "
               "The thick vertical segments show the membership interval [LMF, UMF] at the current input.")
    show(mf_plots.plot_temperature(ctrl.temperature, res.temp_error))
    show(mf_plots.plot_humidity(ctrl.humidity, res.humidity))
    show(mf_plots.plot_output(ctrl.output, res.crisp))
    m1, m2 = st.columns(2)
    for col, d, name in ((m1, res.temp_memberships, "Temperature error"), (m2, res.humidity_memberships, "Humidity")):
        col.markdown(f"**{name}: current membership intervals**")
        col.dataframe(pd.DataFrame([{"Term": k, "LMF": round(v[0], 3), "UMF": round(v[1], 3),
                                     "FOU width": round(v[1] - v[0], 3)} for k, v in d.items()]),
                      hide_index=True, width="stretch")

# ============================================================================
# TAB 3: Rule base
# ============================================================================
def tab_rules():
    st.caption("20 rules = 5 temperature-error terms × 4 humidity terms. The same list is used by the inference engine.")
    df_rules = rules_to_dataframe()
    df_rules["f lower"] = np.round(res.f_lower, 3)
    df_rules["f upper"] = np.round(res.f_upper, 3)
    df_rules["Active"] = np.where(res.f_upper > 0, "✔", "")
    show_only = st.toggle("Show only active rules", value=False)
    view = df_rules[df_rules["Active"] == "✔"] if show_only else df_rules
    st.dataframe(view.style.apply(lambda r: ["background-color:#12a59426" if r["Active"] == "✔" else "" for _ in r], axis=1),
                 hide_index=True, width="stretch", height=min(760, 38 * (len(view) + 1)))
    st.markdown("**Rule matrix** (rows: temperature error, columns: humidity; ✔ = active)")
    mat = pd.DataFrame(index=cfg.RULE_TEMP_ORDER, columns=cfg.RULE_HUM_ORDER, dtype=object)
    for r, fu in zip(RULES, res.f_upper):
        mat.loc[r.temperature_term, r.humidity_term] = f"{r.output_term}" + (f"  ✔ {fu:.2f}" if fu > 0 else "")
    st.dataframe(mat, width="stretch")

# ============================================================================
# TAB 4: Inference
# ============================================================================
def tab_inference():
    st.markdown("### 1. Inputs")
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("T_room", f"{res.t_room:.1f} °C"); i2.metric("T_set", f"{res.t_set:.1f} °C")
    i3.metric("e = T_room − T_set", f"{res.temp_error_raw:+.1f} °C"); i4.metric("Humidity", f"{res.humidity:.0f} %")
    st.markdown("### 2. Fuzzification (intervals [LMF, UMF])")
    f1, f2 = st.columns(2)
    for col, d in ((f1, res.temp_memberships), (f2, res.humidity_memberships)):
        col.dataframe(pd.DataFrame([{"Term": k, "[LMF, UMF]": f"[{v[0]:.3f}, {v[1]:.3f}]"} for k, v in d.items()]),
                      hide_index=True, width="stretch")
    st.markdown("### 3. Rule firing – interval firing strengths")
    st.latex(r"f_l = \min(\underline{\mu}_{T_l}(e),\,\underline{\mu}_{H_l}(h)),\qquad "
             r"F_l = \min(\overline{\mu}_{T_l}(e),\,\overline{\mu}_{H_l}(h))")
    c_l, c_r = ctrl.engine.rule_centroids
    show_rows = [{"Rule": f"R{a.rule.rule_id}", "Consequent": a.rule.output_term,
                  "f (lower)": round(a.f_lower, 3), "F (upper)": round(a.f_upper, 3),
                  "centroid [c_l, c_r]": f"[{c_l[i]:.1f}, {c_r[i]:.1f}]"}
                 for i, a in enumerate(res.activations) if a.active]
    st.dataframe(pd.DataFrame(show_rows), hide_index=True, width="stretch")
    show(inf_plots.plot_firing_strengths(res))
    st.markdown("### 4. Aggregation (max of clipped consequents) and type reduction (Karnik–Mendel)")
    show(inf_plots.plot_aggregation(res))
    tr = res.type_reduction
    if tr is not None:
        k1, k2, k3 = st.columns(3)
        k1.metric("y_left", f"{tr.y_left:.2f}", help=f"switch point L = {tr.left.switch_point}, iterations = {tr.left.iterations}")
        k2.metric("y_right", f"{tr.y_right:.2f}", help=f"switch point R = {tr.right.switch_point}, iterations = {tr.right.iterations}")
        k3.metric("crisp u", f"{res.crisp:+.2f}")
        st.caption(f"KM: left endpoint L = {tr.left.switch_point} ({tr.left.iterations} iterations), "
                   f"right endpoint R = {tr.right.switch_point} ({tr.right.iterations} iterations); "
                   f"converged: {tr.left.converged and tr.right.converged}.")
        st.latex(r"y_l=\min_{f_i\in[f_l,F_l]}\frac{\sum f_i\,c^{l}_i}{\sum f_i},\quad "
                 r"y_r=\max_{f_i\in[f_l,F_l]}\frac{\sum f_i\,c^{r}_i}{\sum f_i},\quad "
                 r"u=\frac{y_l+y_r}{2}")
    else:
        st.warning("No rule fired – output set to 0 (safe neutral).")

# ============================================================================
# TAB 5: Simulation
# ============================================================================
def tab_simulation():
    st.caption("Simplified one-zone thermal model: dT/dt = (T_out − T)/τ + P(u). u < 0 heats, u > 0 cools.")
    a1, a2, a3 = st.columns(3)
    a1.slider("Initial temperature [°C]", *cfg.SIM_LIMITS["t0"], step=0.5, key="sim_t0")
    a2.slider("Set temperature [°C]", *cfg.SIM_LIMITS["t_set"], step=0.5, key="sim_t_set")
    a3.slider("Outdoor temperature [°C]", *cfg.SIM_LIMITS["t_out"], step=0.5, key="sim_t_out")
    b1, b2, b3 = st.columns(3)
    b1.slider("Humidity (constant) [%]", 0.0, 100.0, step=1.0, key="sim_humidity")
    b2.slider("Duration [min]", *cfg.SIM_LIMITS["duration_min"], step=10.0, key="sim_duration")
    b3.slider("Time step [min]", *cfg.SIM_LIMITS["dt_min"], step=0.1, key="sim_dt")
    with st.expander("Model parameters"):
        p1, p2, p3, p4 = st.columns(4)
        p1.slider("Thermal time constant τ [min]", *cfg.SIM_LIMITS["tau_min"], step=10.0, key="sim_tau")
        p2.slider("Max heating rate [°C/min]", *cfg.SIM_LIMITS["heat_rate"], step=0.01, key="sim_heat")
        p3.slider("Max cooling rate [°C/min]", *cfg.SIM_LIMITS["cool_rate"], step=0.01, key="sim_cool")
        p4.slider("Disturbance σ [°C/step]", *cfg.SIM_LIMITS["noise_std"], step=0.01, key="sim_noise",
                  help="0 = fully deterministic; otherwise seeded Gaussian noise.")

    params, fuzzy_df, onoff_df, sim_error = get_sim()
    if sim_error:
        st.error(f"Invalid simulation parameters: {sim_error}")
    else:
        band = float(st.session_state.comfort_band)
        show(sim_plots.plot_temperature(fuzzy_df, band, "IT2FLS – room temperature"))
        show(sim_plots.plot_control(fuzzy_df, "IT2FLS – control signal"))
        mt = compute_metrics(fuzzy_df, params["dt_min"], band)
        st.dataframe(pd.DataFrame({"IT2FLS": fmt_metrics(mt)}), width="stretch")
        st.caption("'—' = the comfort zone was not reached / the temperature did not settle in it. "
                   "Energy proxy is a normalised control-effort index, not kWh.")

# ============================================================================
# TAB 6: Comparison
# ============================================================================
def tab_comparison():
    params, fuzzy_df, onoff_df, sim_error = get_sim()
    if sim_error:
        st.error(f"Invalid simulation parameters: {sim_error}")
    else:
        band = float(st.session_state.comfort_band)
        st.caption(f"Same room and scenario for both controllers (set in the Simulation tab): T0 = {params['t0']:g} °C, "
                   f"T_set = {params['t_set']:g} °C, T_out = {params['t_out']:g} °C, humidity = {params['humidity']:g} %, "
                   f"ON/OFF hysteresis = {st.session_state.hysteresis:g} °C.")
        runs = {"IT2FLS": fuzzy_df, "ON/OFF": onoff_df}
        show(cmp_plots.plot_temperature_comparison(runs, band))
        show(cmp_plots.plot_control_comparison(runs))
        table = pd.DataFrame({n: fmt_metrics(compute_metrics(d, params["dt_min"], band)) for n, d in runs.items()})
        st.dataframe(table, width="stretch")
        st.caption("ON/OFF switches at full power (|u| = 100); IT2FLS modulates intensity. As a proportional-type "
                   "controller IT2FLS has a small steady-state offset, which ON/OFF does not have – at the price of "
                   "constant switching. 'Energy proxy' = Σ|u|/100·Δt (full-power minutes), not kWh.")

# ============================================================================
# TAB 7: Settings
# ============================================================================
def tab_settings():
    s1, s2, s3 = st.columns(3)
    s1.slider("FOU width factor", *cfg.FOU_FACTOR_RANGE, step=0.1, key="fou_factor",
              help="0 = type-1 system (UMF = LMF). Larger values widen the Footprint of Uncertainty.")
    s2.slider("ON/OFF hysteresis [°C]", *cfg.SIM_LIMITS["hysteresis"], step=0.1, key="hysteresis")
    s3.slider("Comfort zone ± [°C] (metrics)", *cfg.SIM_LIMITS["comfort_band"], step=0.1, key="comfort_band")
    st.button("Reset all settings to defaults", on_click=reset_settings)
    st.caption("Thermal-model parameters (τ, heating/cooling strength, outdoor temperature, disturbance) are in the Simulation tab.")
    st.markdown("#### Impact of uncertainty (FOU) on the controller response")
    show(cmp_plots.plot_fou_impact(cached_fou_impact(float(st.session_state.humidity))))
    st.caption("FOU factor 0 gives a degenerate interval (type-1 behaviour); wider FOU widens [y_left, y_right] while "
               "the crisp response stays close – uncertainty mostly softens the transitions between rules.")


# ----------------------------------------------------------------------------
# Renderowanie zakładek
# ----------------------------------------------------------------------------
for _tab, _render in zip(tabs, (tab_dashboard, tab_membership, tab_rules, tab_inference,
                                tab_simulation, tab_comparison, tab_settings)):
    if is_open(_tab):
        with _tab:
            _render()
