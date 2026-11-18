"""Generuje rysunki do dokumentacji (docs/img) z prawdziwego silnika IT2FLS.

Uruchomienie:  python scripts/generate_figures.py
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classical.on_off_controller import OnOffController  # noqa: E402
from config import default_config as cfg  # noqa: E402
from core.analysis import control_surface, fou_impact, response_curve  # noqa: E402
from core.controller import IT2FuzzyController  # noqa: E402
from core.metrics import compute_metrics  # noqa: E402
from core.room_simulation import RoomParams, simulate  # noqa: E402
from visualization import comparison_plots as cp  # noqa: E402
from visualization import inference_plots as ip  # noqa: E402
from visualization import membership_plots as mp  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / name, dpi=110, bbox_inches="tight")
    print("saved", name)


def main():
    c = IT2FuzzyController(cfg.DEFAULT_FOU_FACTOR)
    r = c.compute(20.5, 22.0, 60.0)
    save(mp.plot_temperature(c.temperature, r.temp_error), "membership_temperature.png")
    save(mp.plot_humidity(c.humidity, r.humidity), "membership_humidity.png")
    save(mp.plot_output(c.output, r.crisp), "membership_output.png")
    save(ip.plot_aggregation(r), "aggregation_km.png")
    save(ip.plot_response_curve(response_curve(c, np.linspace(-10, 10, 81), 60), r.temp_error, 60), "response_curve.png")
    e, h = np.linspace(-10, 10, 41), np.linspace(0, 100, 21)
    save(ip.plot_heatmap(e, h, control_surface(c, e, h), (r.temp_error, r.humidity)), "heatmap.png")
    p = RoomParams()
    runs = {"IT2FLS": simulate(c, p).data, "ON/OFF": simulate(OnOffController(cfg.SIM_DEFAULTS["hysteresis"]), p).data}
    save(cp.plot_temperature_comparison(runs), "comparison_temperature.png")
    save(cp.plot_control_comparison(runs), "comparison_control.png")
    save(cp.plot_fou_impact(fou_impact([0, 0.5, 1, 2, 3], np.linspace(-6, 6, 49), 50)), "fou_impact.png")
    for name, df in runs.items():
        print(name, {k: (None if v is None else round(v, 2)) for k, v in compute_metrics(df, p.dt_min, cfg.SIM_DEFAULTS["comfort_band"]).items()})


if __name__ == "__main__":
    main()
