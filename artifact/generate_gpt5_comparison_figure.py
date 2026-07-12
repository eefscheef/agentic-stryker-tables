#!/usr/bin/env python3
"""Ergonomic dumbbell figure for the GPT-Turbo vs GPT-5.4 TestPilot comparison.

Reads the two auto-generated comparison tables verbatim:
  tables/tab-gpt5-stmt-cov-comparison.tex
  tables/tab-gpt5-testpilot-mutation-comparison.tex

and renders, per metric, a dumbbell (connected-dot) plot: one row per package,
a gray GPT-Turbo baseline dot and the two GPT-5.4 dots (500 / 1000 token budget),
connected by a light segment. Direction (improve/regress) and rough magnitude
are readable at a glance; exact per-package deltas are left to the prose.

Output (under figures/):
  gpt5_testpilot_comparison_dumbbell.{pdf,png}
  gpt5_coverage_vs_mutation_scatter.{pdf,png}  -- overlay scatter, same 3 series
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from dumbbell_summary import add_mean_median_lines, summary_legend_handles

HERE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(HERE, "tables")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

C_TURBO = "#9e9e9e"
C_500 = "#74c476"
C_1000 = "#1b6e3a"
C_DAGGER = "#b30000"

# Daggers come straight from the .tex tables (the single source of truth), so the figure
# always matches them. The tables already flag only fs-extra (coverage) and crawler-url-parser.
S_CIRCLE = 46   # GPT-5.4-500 circle (drawn in front)
S_DIAMOND = 72  # GPT-5.4-1000 diamond (drawn behind, deliberately larger so its
                # corners stay visible when the two GPT-5.4 budgets tie and the
                # circle sits on top -- a tie then reads as a green diamond with a
                # lighter circle inside, rather than the diamond vanishing.

# Libraries kept out of the comparison figure (excluded thesis-wide for now; the
# exclusion rationale stays in the text). fs-extra is coverage-only, so dropping
# it removes the last daggered row and the dagger legend entry auto-disappears.
EXCLUDE_LIBS = {"fs-extra"}

PCT = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*\\?%")


def clean_name(cell):
    name = cell.replace(r"$^{\dagger}$", "").replace(r"\dagger", "")
    name = re.sub(r"\\textbf\{([^}]*)\}", r"\1", name)
    return name.strip()


def val(cell):
    m = PCT.search(cell)
    return float(m.group(1)) if m else None


def parse_table(path):
    """Return list of {name, turbo, s500, s1000, dagger} from a comparison .tex."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.endswith(r"\\"):
                continue
            if "Median" in line or "multicolumn" in line or "textbf{Project" in line:
                continue
            cells = [c.strip() for c in line[:-2].split("&")]
            if len(cells) < 6:
                continue
            name = clean_name(cells[0])
            if not name or name.startswith("\\"):
                continue
            if name in EXCLUDE_LIBS:
                continue
            turbo, s500, s1000 = val(cells[1]), val(cells[2]), val(cells[4])
            if turbo is None or s500 is None or s1000 is None:
                continue
            rows.append(dict(name=name, turbo=turbo, s500=s500, s1000=s1000,
                             dagger="dagger" in cells[0]))
    return rows


def dumbbell(ax, rows, xlabel, title):
    rows = sorted(rows, key=lambda r: r["turbo"])  # weakest baseline at bottom
    for i, r in enumerate(rows):
        vals = [r["turbo"], r["s500"], r["s1000"]]
        ax.plot([min(vals), max(vals)], [i, i], color="0.82", lw=2.4, zorder=1,
                solid_capstyle="round")
        # Draw order back-to-front: baseline, then the 1000 diamond, then the 500 circle.
        # The circle sits in front; the larger diamond's corners peek out from behind it.
        ax.scatter(r["turbo"], i, color=C_TURBO, s=42, zorder=3,
                   edgecolor="black", linewidths=0.3)
        ax.scatter(r["s1000"], i, color=C_1000, marker="D", s=S_DIAMOND, zorder=4,
                   edgecolor="black", linewidths=0.3)
        ax.scatter(r["s500"], i, color=C_500, marker="o", s=S_CIRCLE, zorder=5,
                   edgecolor="black", linewidths=0.3)
    add_mean_median_lines(ax, [
        ([r["turbo"] for r in rows], C_TURBO),
        ([r["s500"] for r in rows], C_500),
        ([r["s1000"] for r in rows], C_1000),
    ], mean_style="caret")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["name"] + ("  †" if r["dagger"] else "") for r in rows],
                       fontsize=8.5)
    for tick, r in zip(ax.get_yticklabels(), rows):
        if r["dagger"]:
            tick.set_color(C_DAGGER)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(-2, 102)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", ls=":", alpha=0.5)
    ax.set_axisbelow(True)


def plot_scatter_overlay(cov, mut):
    """Overlay scatter (x = statement coverage, y = mutation score), one point per
    library, for GPT-Turbo / GPT-5.4-500 / GPT-5.4-1000. Joined from the same two
    tables as the dumbbell, so the numbers match exactly. No y=x reference line."""
    cov_by = {r["name"]: r for r in cov}
    mut_by = {r["name"]: r for r in mut}
    libs = sorted(set(cov_by) & set(mut_by))
    series = [
        ("GPT-Turbo", "turbo", "#3b6fb0", "o"),
        ("GPT-5.4 · 500 tok", "s500", C_500, "s"),
        ("GPT-5.4 · 1000 tok", "s1000", C_1000, "D"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    for label, field, color, marker in series:
        xs = [cov_by[l][field] for l in libs]
        ys = [mut_by[l][field] for l in libs]
        ax.scatter(xs, ys, c=color, marker=marker, s=52, edgecolors="black",
                   linewidths=0.4, alpha=0.85, label=f"{label}  (n={len(libs)})")
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_aspect("equal")
    ax.set_xlabel("Statement coverage (%)")
    ax.set_ylabel("Mutation score (%)")
    ax.set_title("Coverage vs. mutation score per library")
    ax.grid(True, ls=":", alpha=0.45)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.92)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(FIG, f"gpt5_coverage_vs_mutation_scatter.{ext}")
        fig.savefig(out, dpi=150)
        print(f"wrote {out}")
    plt.close(fig)


def main():
    cov = parse_table(os.path.join(TBL, "tab-gpt5-stmt-cov-comparison.tex"))
    mut = parse_table(os.path.join(TBL, "tab-gpt5-testpilot-mutation-comparison.tex"))
    print(f"parsed coverage rows: {len(cov)}, mutation rows: {len(mut)}")

    fig, (axc, axm) = plt.subplots(1, 2, figsize=(13, 7.6))
    dumbbell(axc, cov, "Statement coverage (%)", "Statement coverage")
    dumbbell(axm, mut, "Mutation score (%)", "Mutation score")

    handles = [
        Line2D([], [], marker="o", color="none", markerfacecolor=C_TURBO,
               markeredgecolor="black", markersize=7, label="GPT-Turbo (baseline)"),
        Line2D([], [], marker="o", color="none", markerfacecolor=C_500,
               markeredgecolor="black", markersize=7, label="GPT-5.4 · 500 tok"),
        Line2D([], [], marker="D", color="none", markerfacecolor=C_1000,
               markeredgecolor="black", markersize=7, label="GPT-5.4 · 1000 tok"),
    ]
    # Only advertise the dagger if some plotted row still carries one.
    if any(r["dagger"] for r in cov) or any(r["dagger"] for r in mut):
        handles.append(
            Line2D([], [], marker="s", color="none", markerfacecolor="white",
                   markeredgecolor="white", markersize=0,
                   label="†  problematic replication/validation case"))
    handles += summary_legend_handles(mean_style="caret")
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("GPT-Turbo vs. GPT-5.4 TestPilot — coverage and mutation score per package",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    for ext in ("pdf", "png"):
        out = os.path.join(FIG, f"gpt5_testpilot_comparison_dumbbell.{ext}")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)

    plot_scatter_overlay(cov, mut)


if __name__ == "__main__":
    main()
