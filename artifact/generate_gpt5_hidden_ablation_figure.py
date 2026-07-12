#!/usr/bin/env python3
"""Dumbbell figure for the GPT-5.4 hidden mutation-feedback ablation.

Reads the auto-generated CSV (single source of truth, so the figure can never
drift from tab-gpt5-agent-hidden-ablation-scores.tex):
  tables/tab-gpt5-agent-hidden-ablation.csv

Each row is one benchmark package. A gray dot marks the Hidden (ablated)
condition's mutation score and a blue dot the Full-feedback pre-residual score;
the connecting segment is colored by direction (green = full feedback helps,
red = full feedback hurts). This is the dumbbell view of the Delta_feedback
column. The comparison is UNPAIRED (Hidden and Full are independent runs;
the segment connects two condition means, not a before/after on one run).

Rows are sorted by Delta_feedback, so the few large outliers fall to the
extremes and the near-zero bulk in the middle shows the small, mixed effect.
Packages whose mean used only one run are marked '*' (matching the table).
uneval's Full value is 0.0% because the full-feedback runs generated no usable
suite (a generation failure, not a measured zero) -- discussed in the text.

Output (under figures/):
  gpt5_hidden_ablation_dumbbell.{pdf,png}
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

CSV = os.path.join(TBL, "tab-gpt5-agent-hidden-ablation.csv")

# Dropped thesis-wide for now (matches the table generators).
EXCLUDE_LIBS = {"fs-extra"}

C_HIDDEN = "#9e9e9e"   # ablated baseline (no per-item feedback)
C_FULL = "#2166ac"     # full feedback, pre-residual
C_UP = "#2e7d32"       # full feedback helps (Delta > 0)
C_DOWN = "#c62828"     # full feedback hurts (Delta < 0)
C_FLAT = "#bdbdbd"     # |Delta| ~ 0
FLAT_EPS = 0.5         # pp band treated as "no change" for segment color


def fnum(s):
    s = (s or "").strip()
    if s == "" or s.lower() == "none":
        return None
    return float(s)


def load_rows():
    rows = []
    with open(CSV) as fh:
        for r in csv.DictReader(fh):
            if r["library"] in EXCLUDE_LIBS:
                continue
            hidden = fnum(r["hidden_score"])
            full = fnum(r["full_pre_score"])
            if hidden is None or full is None:
                continue
            hn = fnum(r["hidden_score_n"])
            fn = fnum(r["full_pre_n"])
            single = (hn is not None and hn < 2) or (fn is not None and fn < 2)
            rows.append({
                "name": r["library"],
                "hidden": hidden,
                "full": full,
                "delta": fnum(r["delta_feedback"]),
                "single": single,
            })
    return rows


def seg_color(delta):
    if delta is None or abs(delta) <= FLAT_EPS:
        return C_FLAT
    return C_UP if delta > 0 else C_DOWN


def main():
    rows = sorted(load_rows(), key=lambda r: (r["delta"] if r["delta"] is not None else 0.0))

    fig, ax = plt.subplots(figsize=(8.2, 8.6))
    for i, r in enumerate(rows):
        ax.plot([r["hidden"], r["full"]], [i, i], color=seg_color(r["delta"]),
                lw=3.0, zorder=2, solid_capstyle="round", alpha=0.9)
        ax.scatter(r["hidden"], i, color=C_HIDDEN, s=46, zorder=3,
                   edgecolor="black", linewidths=0.3)
        ax.scatter(r["full"], i, color=C_FULL, s=46, zorder=4,
                   edgecolor="black", linewidths=0.3)

    add_mean_median_lines(ax, [
        ([r["hidden"] for r in rows], C_HIDDEN),
        ([r["full"] for r in rows], C_FULL),
    ], mean_style="caret")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["name"] + ("  *" if r["single"] else "") for r in rows],
                       fontsize=9)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(-2, 102)
    ax.set_xlabel("Mutation score (%)")
    ax.set_title("Hidden vs. full mutation feedback (pre-residual) per package",
                 fontsize=12)
    ax.grid(axis="x", ls=":", alpha=0.5)
    ax.set_axisbelow(True)

    handles = [
        Line2D([], [], marker="o", color="none", markerfacecolor=C_HIDDEN,
               markeredgecolor="black", markersize=8, label="Hidden feedback"),
        Line2D([], [], marker="o", color="none", markerfacecolor=C_FULL,
               markeredgecolor="black", markersize=8,
               label="Full feedback (pre-residual)"),
        Line2D([], [], color=C_UP, lw=3, label="Full feedback helps ($\\Delta>0$)"),
        Line2D([], [], color=C_DOWN, lw=3, label="Full feedback hurts ($\\Delta<0$)"),
        Line2D([], [], marker="$*$", color="none", markerfacecolor="black",
               markeredgecolor="none", markersize=10, label="mean from a single run"),
    ] + summary_legend_handles(mean_style="caret")
    # Below the axes so it never covers a marker.
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("pdf", "png"):
        out = os.path.join(FIG, f"gpt5_hidden_ablation_dumbbell.{ext}")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"wrote {os.path.relpath(out, HERE)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
