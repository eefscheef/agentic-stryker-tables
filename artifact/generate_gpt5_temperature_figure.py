#!/usr/bin/env python3
"""Dumbbell figure for the GPT-5.4 temperature comparison (final mutation scores).

Reads tables/tab-gpt5-agent-temperature.csv (produced by
generate_gpt5_agent_temperature_table.py) and plots, per benchmark library, the
final (post-residual) mutation score at temperature 1 (mean of R3/R4) and at
temperature 0, connected by a segment. The segment's direction/length is the
per-library temperature effect; the headline is that there is no systematic one
(the segments go both ways and the paired median is ~0).

Markings:
  * graceful-fs's temperature-1 value is a single-run mean (suffix '*').
  * uneval is red: both of its temperature-1 runs generated no tests, so its
    temperature-1 final is 0.0% and its large segment is an artifact, not a
    temperature effect. crawler-url-parser is NOT flagged even though the table
    generator lists it in temp1_no_test_libs (>=1 no-test run): only one of its
    two temperature-1 runs failed, so its plotted value (29.9%) is a real mean
    of a 0.0% run and a 59.7% run, and its segment is ordinary-sized.

Output (under figures/):
  gpt5_temperature_dumbbell.{pdf,png}
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from dumbbell_summary import add_mean_median_lines, summary_legend_handles

HERE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(HERE, "tables")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

CSV = os.path.join(TBL, "tab-gpt5-agent-temperature.csv")

EXCLUDE_LIBS = {"fs-extra"}            # excluded thesis-wide
NO_TEST_T1 = {"uneval"}  # only libs whose temp-1 mean is entirely a no-test failure (both runs 0)

C_T1 = "#9e9e9e"    # temperature 1 (mean of R3/R4), the established runs
C_T0 = "#1f78b4"    # temperature 0
C_UP = "#2e7d32"    # temperature 0 higher
C_DOWN = "#c62828"  # temperature 1 higher
C_FLAT = "#bdbdbd"
C_FLAG = "#b30000"  # no-test-artifact label colour
FLAT_EPS = 0.5      # |Delta| <= this pp is treated as "no change" for the segment colour


def fnum(s):
    s = (s or "").strip()
    return float(s) if s not in ("", "None") else None


def load_rows():
    rows = []
    with open(CSV) as fh:
        for r in csv.DictReader(fh):
            lib = r["library"]
            if lib in EXCLUDE_LIBS:
                continue
            t1 = fnum(r["final_score_temp1_mean"])
            t0 = fnum(r["final_score_temp0"])
            if t1 is None or t0 is None:
                continue
            n = fnum(r["final_score_temp1_n"])
            rows.append(dict(name=lib, t1=t1, t0=t0, delta=t0 - t1,
                             single=(n is not None and n < 2),
                             notest=lib in NO_TEST_T1))
    return rows


def seg_color(d):
    if abs(d) <= FLAT_EPS:
        return C_FLAT
    return C_UP if d > 0 else C_DOWN


def main():
    rows = sorted(load_rows(), key=lambda r: r["delta"])

    fig, ax = plt.subplots(figsize=(8.2, 8.4))
    for i, r in enumerate(rows):
        ax.plot([r["t1"], r["t0"]], [i, i], color=seg_color(r["delta"]),
                lw=3.0, zorder=2, solid_capstyle="round", alpha=0.9)
        ax.scatter(r["t1"], i, color=C_T1, s=46, zorder=3, edgecolor="black", linewidths=0.3)
        ax.scatter(r["t0"], i, color=C_T0, s=46, zorder=4, edgecolor="black", linewidths=0.3)

    add_mean_median_lines(ax, [
        ([r["t1"] for r in rows], C_T1),
        ([r["t0"] for r in rows], C_T0),
    ], mean_style="caret")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["name"] + ("  *" if r["single"] else "") for r in rows], fontsize=9)
    for tick, r in zip(ax.get_yticklabels(), rows):
        if r["notest"]:
            tick.set_color(C_FLAG)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(-2, 102)
    ax.set_xlabel("Final mutation score (%)")
    ax.set_title("Temperature 0 vs. temperature 1 final mutation score per library", fontsize=12)
    ax.grid(axis="x", ls=":", alpha=0.5)
    ax.set_axisbelow(True)

    handles = [
        Line2D([], [], marker="o", color="none", markerfacecolor=C_T1,
               markeredgecolor="black", markersize=8, label="Temperature 1"),
        Line2D([], [], marker="o", color="none", markerfacecolor=C_T0,
               markeredgecolor="black", markersize=8, label="Temperature 0"),
        Line2D([], [], color=C_UP, lw=3, label="Temp. 0 higher"),
        Line2D([], [], color=C_DOWN, lw=3, label="Temp. 1 higher"),
        Line2D([], [], marker="$*$", color="black", linestyle="None",
               markersize=10, label="single T1 run"),
    ] + summary_legend_handles(mean_style="caret")
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("pdf", "png"):
        out = os.path.join(FIG, f"gpt5_temperature_dumbbell.{ext}")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"wrote {os.path.relpath(out, HERE)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
