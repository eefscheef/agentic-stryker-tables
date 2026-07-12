#!/usr/bin/env python3
"""Per-library dumbbell figure for the Agent vs. TestPilot comparison.

Reads the auto-generated per-library CSV (single source of truth, so the
figure cannot drift from tab-gpt5-agent-testpilot-comparison.tex, whose
aggregate rows are the means/medians of exactly these values):
  tables/tab-gpt5-agent-testpilot-per-library.csv

Two panels: mutation score and statement coverage. Each row is one benchmark
library with three markers -- TestPilot-500 (light green circle),
TestPilot-1000 (dark green diamond) and the final agent (purple square) --
connected by a neutral segment. Rows are sorted per panel by the agent's
delta over TestPilot-1000, so agent wins collect at the top and losses at the
bottom. Draw order is diamond (largest, back), agent square, then the 500
circle in front, so ties stay visible as concentric shapes.

uneval's label is red: the agent produced no tests for it in either
temperature-1 run, so its agent values are 0.0% by convention (a generation
failure, not a measured zero) -- discussed in the text.

Per-series summary marks come from dumbbell_summary and land on the aggregate
table's Mut./Cov. cells (validated in the console output). With three series,
the means are drawn as carets on the top edge rather than full-height lines so
only the three median lines cross the data.

Output (under figures/):
  gpt5_agent_testpilot_dumbbell.{pdf,png}
"""
import csv
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from dumbbell_summary import add_mean_median_lines, summary_legend_handles

HERE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(HERE, "tables")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

CSV = os.path.join(TBL, "tab-gpt5-agent-testpilot-per-library.csv")

C_500 = "#74c476"    # same identities as the GPT-Turbo comparison dumbbell
C_1000 = "#1b6e3a"
C_AGENT = "#6a51a3"  # new entity, validated against the two greens
C_FLAG = "#b30000"   # no-test-artifact label colour

NO_TEST_AGENT = {"uneval"}  # agent produced no tests in either temp-1 run

S_DIAMOND = 72  # back
S_AGENT = 54    # middle
S_CIRCLE = 40   # front; strictly smaller so all three stay visible on ties


def load_rows():
    rows = []
    with open(CSV) as fh:
        for r in csv.DictReader(fh):
            rows.append(dict(
                name=r["library"],
                mut=dict(s500=float(r["tp500_mut"]), s1000=float(r["tp1000_mut"]),
                         agent=float(r["agent_mut"])),
                cov=dict(s500=float(r["tp500_stmt"]), s1000=float(r["tp1000_stmt"]),
                         agent=float(r["agent_stmt"])),
            ))
    return rows


def dumbbell(ax, rows, metric, xlabel, title):
    rows = sorted(rows, key=lambda r: r[metric]["agent"] - r[metric]["s1000"])
    for i, r in enumerate(rows):
        v = r[metric]
        ax.plot([min(v.values()), max(v.values())], [i, i], color="0.82", lw=2.4,
                zorder=1, solid_capstyle="round")
        ax.scatter(v["s1000"], i, color=C_1000, marker="D", s=S_DIAMOND, zorder=3,
                   edgecolor="black", linewidths=0.3)
        ax.scatter(v["agent"], i, color=C_AGENT, marker="s", s=S_AGENT, zorder=4,
                   edgecolor="black", linewidths=0.3)
        ax.scatter(v["s500"], i, color=C_500, marker="o", s=S_CIRCLE, zorder=5,
                   edgecolor="black", linewidths=0.3)
    # Three series: full-height lines only for the medians; means become carets
    # on the top edge so the panel doesn't fill up with six vertical lines.
    summary = add_mean_median_lines(ax, [
        ([r[metric]["s500"] for r in rows], C_500),
        ([r[metric]["s1000"] for r in rows], C_1000),
        ([r[metric]["agent"] for r in rows], C_AGENT),
    ], mean_style="caret")
    for label, color in (("TestPilot-500", C_500), ("TestPilot-1000", C_1000),
                         ("Agent", C_AGENT)):
        s = summary[color]
        print(f"  {title:>18} {label:<15} mean={s['mean']:.1f}  median={s['median']:.1f}")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["name"] for r in rows], fontsize=8.5)
    for tick, r in zip(ax.get_yticklabels(), rows):
        if r["name"] in NO_TEST_AGENT:
            tick.set_color(C_FLAG)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(-2, 102)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", ls=":", alpha=0.5)
    ax.set_axisbelow(True)


def main():
    rows = load_rows()
    print(f"parsed {len(rows)} libraries")

    fig, (axm, axc) = plt.subplots(1, 2, figsize=(13, 7.6))
    dumbbell(axm, rows, "mut", "Mutation score (%)", "Mutation score")
    dumbbell(axc, rows, "cov", "Statement coverage (%)", "Statement coverage")

    handles = [
        Line2D([], [], marker="o", color="none", markerfacecolor=C_500,
               markeredgecolor="black", markersize=6.5, label="TestPilot-500"),
        Line2D([], [], marker="D", color="none", markerfacecolor=C_1000,
               markeredgecolor="black", markersize=7, label="TestPilot-1000"),
        Line2D([], [], marker="s", color="none", markerfacecolor=C_AGENT,
               markeredgecolor="black", markersize=7.5, label="Agent (final)"),
    ] + summary_legend_handles(mean_style="caret")
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("GPT-5.4 agent vs. GPT-5.4 TestPilot — mutation score and coverage per library",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    for ext in ("pdf", "png"):
        out = os.path.join(FIG, f"gpt5_agent_testpilot_dumbbell.{ext}")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"wrote {os.path.relpath(out, HERE)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
