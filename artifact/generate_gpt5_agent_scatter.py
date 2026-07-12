#!/usr/bin/env python3
"""Section-2 (Agent Results) scatter: the agent's own coverage vs. mutation.

Self-contained, NO TestPilot baseline — one point per benchmark library, where
  x = statement coverage (%), y = final (post-residual) mutation score (%)
for the agent's headline runs (full+residual, temperature 1; mean of R3+R4).

Reuses the exact extraction conventions of the published agent comparison table
(generate_gpt5_agent_testpilot_comparison_table.py). Points cover the 20-library
comparison set (fs-extra is excluded thesis-wide).

Output (under figures/):
  gpt5_agent_coverage_vs_mutation_scatter.{pdf,png}
"""
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import generate_gpt5_agent_residual_table as ar
import generate_gpt5_agent_testpilot_comparison_table as cmp

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

C_AGENT = "#238b45"  # agent green

# Manual per-label horizontal placement (square layout). Default is to the right
# of the marker; libraries listed here sit to the left.
LABEL_LEFT = {"pull-stream", "omnitool", "complex.js", "plural",
              "zip-a-folder", "image-downloader", "jsonfile"}
# Full custom placement for special cases: lib -> (dx, dy, ha, va), offsets in points.
LABEL_CUSTOM = {
    "geo-point": (2, -6, "center", "top"),  # below the marker, nudged up/right off complex.js
}
DX = 7   # horizontal label offset, in points (clears the dot)


def mean(vals):
    return statistics.mean(vals) if vals else None


def agent_stmt_coverage(runs, lib):
    """Mean statement coverage over R3/R4, mirroring collect_agent_coverage:
    no-test runs contribute 0.0, incomplete-measurement runs are skipped."""
    vals = []
    for run_id in ar.RUN_IDS:
        cell = runs[run_id].get(lib)
        if not cell:
            continue
        if cell.get("no_test_score"):
            vals.append(0.0)
            continue
        if not cell.get("score_measurement_complete"):
            continue
        report = ar.read_json(os.path.join(
            ar.DATA_DIR, ar.RUN_DIRS[run_id], lib, "report", "report.json"))
        if report is None:
            continue
        stmt = report.get("coverage", {}).get("total", {}).get("statements", {}).get("pct")
        if isinstance(stmt, (int, float)):
            vals.append(stmt)
    return mean(vals)


def collect():
    rates = ar.load_pricing()
    runs = {r: ar.collect_run(r) for r in ar.RUN_IDS}
    rows = ar.aggregate(runs, rates)
    points = []
    for lib in cmp.AGENT_SCORED_LIBS:  # agent-scored libs; fs-extra excluded below
        if lib in ar.EXCLUDE_LIBS:  # fs-extra dropped thesis-wide
            continue
        mut = rows[lib]["residual_score"]          # final (post-residual) mutation
        cov = agent_stmt_coverage(runs, lib)
        if mut is None or cov is None:
            continue
        single = rows[lib].get("residual_score_n") == 1   # 1-run (starred) value
        points.append((lib, cov, mut, single))
    return points


def main():
    points = collect()
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    print(f"libraries plotted: {len(points)}")
    print(f"  coverage  mean={statistics.mean(xs):.1f}  median={statistics.median(xs):.1f}")
    print(f"  mutation  mean={statistics.mean(ys):.1f}  median={statistics.median(ys):.1f}")

    labels = [p[0] for p in points]
    med_cov = statistics.median(xs)
    med_mut = statistics.median(ys)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axvline(med_cov, color="0.6", ls="--", lw=0.9, zorder=1)
    ax.axhline(med_mut, color="0.6", ls="--", lw=0.9, zorder=1)
    full_pts = [p for p in points if not p[3]]
    hollow_pts = [p for p in points if p[3]]  # single-run (starred) libraries
    ax.scatter([p[1] for p in full_pts], [p[2] for p in full_pts], c=C_AGENT,
               marker="o", s=55, edgecolors="black", linewidths=0.4, alpha=0.85, zorder=4)
    ax.scatter([p[1] for p in hollow_pts], [p[2] for p in hollow_pts], facecolors="none",
               marker="o", s=55, edgecolors=C_AGENT, linewidths=1.5, zorder=4)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_aspect("equal")
    ax.set_xlabel("Statement coverage (%)")
    ax.set_ylabel("Mutation score (%)")
    ax.set_title("GPT-5.4 agent: coverage vs. mutation score per library")
    ax.grid(True, ls=":", alpha=0.4)

    for lib, cov, mut, single in points:
        if lib in LABEL_CUSTOM:
            dx, dy, ha, va = LABEL_CUSTOM[lib]
            ax.annotate(lib, (cov, mut), textcoords="offset points", xytext=(dx, dy),
                        ha=ha, va=va, fontsize=9.5, color="0.15", zorder=5)
            continue
        right = lib not in LABEL_LEFT
        ax.annotate(lib, (cov, mut), textcoords="offset points",
                    xytext=(DX if right else -DX, 0),
                    ha="left" if right else "right", va="center",
                    fontsize=9.5, color="0.15", zorder=5)

    ax.text(0.035, 0.965,
            f"n = {len(points)}\nmedian cov = {med_cov:.0f}%\nmedian mut = {med_mut:.0f}%",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5, zorder=6,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.95))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(FIG, f"gpt5_agent_coverage_vs_mutation_scatter.{ext}")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
