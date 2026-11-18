"""Wspólny styl wykresów."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HEAT = "#e4572e"
COOL = "#2a6fdb"
NEUTRAL = "#8a8f98"
FUZZY = "#12a594"
ONOFF = "#f29e4c"
SETPOINT = "#222222"
PALETTE = ["#2a6fdb", "#12a594", "#7c5cbf", "#e4572e", "#d9a400", "#c2437a", "#5b7083"]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
        "axes.spines.right": False, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelsize": 10, "legend.fontsize": 8.5, "legend.frameon": False,
        "font.size": 10, "figure.autolayout": False,
    })


apply_style()
