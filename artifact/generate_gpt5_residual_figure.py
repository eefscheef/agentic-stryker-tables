#!/usr/bin/env python3
"""Paired dumbbell figure for the residual-feedback effect (pre -> final).

Reads tables/tab-gpt5-agent-temperature.csv (produced by
generate_gpt5_agent_temperature_table.py; the same source as
tab-gpt5-agent-residual-scores.tex) and plots, per benchmark library, the
agent's mutation score before and after the residual phase -- the dumbbell
view of that table's two Delta columns. Unlike the hidden ablation, this
comparison is genuinely PAIRED: pre and final are the same run measured
before/after residual tests are added.

Two panels with shared rows: temperature 1 (mean of two runs) and temperature 0.
Rows are sorted by the mean residual delta across both temperatures, so the
big residual wins (plural) collect at the top and the no-movers at the
bottom. Segments are green when the residual phase helps and red when it
hurts (graceful-fs's small negatives are Stryker noise, see the text).

Markings (temperature-1 conventions, matching the temperature dumbbell):
  * graceful-fs's temperature-1 values are single-run means (suffix '*').
  * uneval is red: both temperature-1 runs generated no tests, so its
    temperature-1 pre/final are 0.0% (a generation artifact); at temperature 0
    it completed normally at 91.5%.

Per-series mean/median lines come from dumbbell_summary and land on the
Mean/Median rows of tab-gpt5-agent-residual-scores.tex (see console output).

Output (under figures/):
  gpt5_residual_dumbbell.{pdf,png}
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

EXCLUDE_LIBS = {"fs-extra"}   # excluded thesis-wide
NO_TEST_T1 = {"uneval"}       # both temp-1 runs produced no tests (0.0% artifact)

C_PRE = "#9e9e9e"     # pre-residual
C_FINAL = "#6a51a3"   # agent final (same entity colour as the agent-vs-TestPilot figure)
C_UP = "#2e7d32"      # residual phase helps
C_DOWN = "#c62828"    # residual phase hurts
C_FLAT = "#bdbdbd"
C_FLAG = "#b30000"    # no-test-artifact label colour
FLAT_EPS = 0.5        # |Delta| <= this pp is treated as "no change"


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
            vals = dict(
                t1=(fnum(r["pre_score_temp1_mean"]), fnum(r["final_score_temp1_mean"])),
                t0=(fnum(r["pre_score_temp0"]), fnum(r["final_score_temp0"])),
            )
            if any(v is None for pair in vals.values() for v in pair):
                print(f"  skipping {lib}: incomplete pre/final measurements")
                continue
            n1 = fnum(r["pre_score_temp1_n"])
            rows.append(dict(name=lib, **vals,
                             single=(n1 is not None and n1 < 2),
                             notest=lib in NO_TEST_T1))
    return rows


def seg_color(d):
    if abs(d) <= FLAT_EPS:
        return C_FLAT
    return C_UP if d > 0 else C_DOWN


def panel(ax, rows, key, title):
    for i, r in enumerate(rows):
        pre, fin = r[key]
        ax.plot([pre, fin], [i, i], color=seg_color(fin - pre), lw=3.0, zorder=2,
                solid_capstyle="round", alpha=0.9)
        ax.scatter(pre, i, color=C_PRE, s=46, zorder=3, edgecolor="black", linewidths=0.3)
        ax.scatter(fin, i, color=C_FINAL, s=46, zorder=4, edgecolor="black", linewidths=0.3)
    summary = add_mean_median_lines(ax, [
        ([r[key][0] for r in rows], C_PRE),
        ([r[key][1] for r in rows], C_FINAL),
    ], mean_style="caret")
    for label, color in (("pre-residual", C_PRE), ("final", C_FINAL)):
        s = summary[color]
        print(f"  {title:>28} {label:<13} mean={s['mean']:.1f}  median={s['median']:.1f}")
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(-2, 102)
    ax.set_xlabel("Mutation score (%)")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", ls=":", alpha=0.5)
    ax.set_axisbelow(True)


def main():
    # Shared row order across both panels: mean residual delta over the two
    # temperatures, so a library sits at the same height left and right.
    rows = sorted(load_rows(),
                  key=lambda r: ((r["t1"][1] - r["t1"][0]) + (r["t0"][1] - r["t0"][0])) / 2)
    print(f"parsed {len(rows)} libraries")

    fig, (ax1, ax0) = plt.subplots(1, 2, figsize=(11.5, 8.4), sharey=True)
    panel(ax1, rows, "t1", "Temperature 1 (mean of two runs)")
    panel(ax0, rows, "t0", "Temperature 0")

    ax1.set_yticks(range(len(rows)))
    ax1.set_yticklabels([r["name"] + ("  *" if r["single"] else "") for r in rows],
                        fontsize=9)
    for tick, r in zip(ax1.get_yticklabels(), rows):
        if r["notest"]:
            tick.set_color(C_FLAG)

    handles = [
        Line2D([], [], marker="o", color="none", markerfacecolor=C_PRE,
               markeredgecolor="black", markersize=8, label="Pre-residual"),
        Line2D([], [], marker="o", color="none", markerfacecolor=C_FINAL,
               markeredgecolor="black", markersize=8, label="Agent final"),
        Line2D([], [], color=C_UP, lw=3, label="Residual helps ($\\Delta>0$)"),
        Line2D([], [], color=C_DOWN, lw=3, label="Residual hurts ($\\Delta<0$)"),
        Line2D([], [], marker="$*$", color="black", linestyle="None",
               markersize=10, label="single T1 run"),
    ] + summary_legend_handles(mean_style="caret")
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Residual-phase effect on mutation score per library", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    for ext in ("pdf", "png"):
        out = os.path.join(FIG, f"gpt5_residual_dumbbell.{ext}")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"wrote {os.path.relpath(out, HERE)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
