#!/usr/bin/env python3
"""Shared mean/median reference lines for the GPT-5.4 dumbbell figures.

Each dumbbell plots one metric on the x-axis (mutation score or coverage, in %)
with two or three colour-coded series. This draws, per series, a dash-dot
vertical line at the series median and marks the mean, in the series colour.
The house style for the mean is a caret: a small downward triangle on the top
axis edge (mean_style="caret"), so only the median lines cross the data; a
full-height dashed line (mean_style="line") remains available. Mean/median are
computed over exactly the plotted values, so no-test 0.0% entries are included
-- matching the numbers quoted in the prose.

Used by all the generate_gpt5_*_figure.py dumbbell generators so the figures
stay identical.
"""
import statistics as st

from matplotlib.lines import Line2D

# Behind the segments (zorder 2) and markers (>=3), above the grid.
MEAN_STYLE = dict(linestyle="--", linewidth=1.5, alpha=0.8, zorder=1.6)
MEDIAN_STYLE = dict(linestyle="-.", linewidth=1.6, alpha=0.9, zorder=1.7)


def add_mean_median_lines(ax, series, mean_style="line"):
    """Draw per-series vertical mean and median (dash-dot) reference lines.

    series: iterable of (values, color); None values are dropped before the
    mean/median so partially-measured series still summarise their real points.
    mean_style: "caret" (the house style, used by every thesis dumbbell) draws
    the mean as a small downward triangle on the top edge; "line" draws a
    full-height dashed line instead.
    Returns {color: {"mean": m, "median": med}} for optional validation/logging.
    """
    out = {}
    for values, color in series:
        vals = [v for v in values if v is not None]
        if not vals:
            continue
        m = st.mean(vals)
        med = st.median(vals)
        if mean_style == "caret":
            ax.plot([m], [1.0], marker="v", color=color, markeredgecolor="black",
                    markeredgewidth=0.3, markersize=8, linestyle="None",
                    transform=ax.get_xaxis_transform(), clip_on=False, zorder=6)
        else:
            ax.axvline(m, color=color, **MEAN_STYLE)
        ax.axvline(med, color=color, **MEDIAN_STYLE)
        out[color] = {"mean": m, "median": med}
    return out


def summary_legend_handles(mean_style="line"):
    """Two neutral legend entries explaining the summary marks (colour = series)."""
    if mean_style == "caret":
        mean_handle = Line2D([], [], marker="v", color="none", markerfacecolor="0.45",
                             markeredgecolor="black", markersize=7, label="series mean")
    else:
        mean_handle = Line2D([], [], color="0.35", linestyle="--", lw=1.5,
                             label="series mean")
    return [
        mean_handle,
        Line2D([], [], color="0.35", linestyle="-.", lw=1.6, label="series median"),
    ]
