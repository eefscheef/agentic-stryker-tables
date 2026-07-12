#!/usr/bin/env python3
"""Token-usage-vs-project-size scatter for the TestPilot token comparison.

Reads the raw per-library CSV emitted by generate_gpt5_token_table.py
(single source of truth, so the figure cannot drift from
tab-gpt5-testpilot-token-comparison.tex):
  tables/tab-gpt5-token-usage.csv

Two log-log panels sharing the y-axis (total tokens = input + output):
  left  x = LOC  -- "project size" as quoted in the prose; rough scaling
                    with visible outliers (the not-uniform part)
  right x = FUT  -- exported functions under test; TestPilot prompts per
                    function, so this is near-linear and explains the
                    LOC-panel outliers (graceful-fs: 208 LOC but 187 FUT)

Per panel and series a dashed power-law fit (linear in log-log) is drawn and
the Spearman rank correlation is annotated; slopes/rhos are also printed for
use in the prose. GPT-Turbo is drawn hollow because its values are estimated
upper bounds (see the table caption); GPT-5.4 counts are actual API numbers.
The agent columns in the CSV are ignored: this figure accompanies the
TestPilot-only token table.

Output (under figures/):
  gpt5_token_scaling_scatter.{pdf,png}
"""
import csv
import math
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(HERE, "tables")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

CSV = os.path.join(TBL, "tab-gpt5-token-usage.csv")

# Same series identities as the comparison dumbbell.
C_TURBO_EDGE = "#757575"  # darker than the dumbbells' fill gray: hollow markers
                          # on white need the edge itself to carry the contrast
C_500 = "#74c476"
C_1000 = "#1b6e3a"

# Direct labels, all of them libraries the running text names: the size
# extremes (omnitool largest, zip-a-folder smallest; the FUT panel has no
# unambiguous smallest -- uneval and image-downloader tie at FUT=1) plus the
# libraries whose LOC badly under-represents their API surface. The latter get
# their FUT spelled out (value True) so the vertical scatter in the LOC panel
# reads as explained, not as noise.
ANNOTATE_LOC = {"graceful-fs": True, "dirty": True,
                "omnitool": False, "zip-a-folder": False}
ANNOTATE_FUT = {"graceful-fs": False, "omnitool": False}


def load_rows():
    rows = []
    with open(CSV) as fh:
        for r in csv.DictReader(fh):
            def tot(prefix):
                i, o = r.get(f"{prefix}_input"), r.get(f"{prefix}_output")
                if not i or not o:
                    return None
                return int(i) + int(o)
            if not r["loc"] or not r["fut"]:
                continue
            rows.append(dict(name=r["library"], loc=int(r["loc"]), fut=int(r["fut"]),
                             turbo=tot("turbo"), s500=tot("s500"), s1000=tot("s1000")))
    return rows


def _ranks(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(xs, ys):
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den


def fit_loglog(xs, ys):
    """Least-squares fit log10(y) = a + b*log10(x); returns (a, b)."""
    lx = [math.log10(x) for x in xs]
    ly = [math.log10(y) for y in ys]
    mx, my = st.mean(lx), st.mean(ly)
    b = sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / sum((x - mx) ** 2 for x in lx)
    return my - b * mx, b


def panel(ax, rows, xkey, xlabel, title, annotate):
    series = [
        ("GPT-Turbo (est.)", "turbo", dict(marker="o", s=46, facecolors="none",
                                           edgecolors=C_TURBO_EDGE, linewidths=1.3)),
        ("GPT-5.4 · 500 tok", "s500", dict(marker="o", s=46, color=C_500,
                                           edgecolor="black", linewidths=0.3)),
        ("GPT-5.4 · 1000 tok", "s1000", dict(marker="D", s=40, color=C_1000,
                                             edgecolor="black", linewidths=0.3)),
    ]
    stats_lines = []
    for label, field, style in series:
        pts = [(r[xkey], r[field]) for r in rows if r[field] is not None]
        xs, ys = zip(*pts)
        line_color = C_TURBO_EDGE if field == "turbo" else style["color"]
        a, b = fit_loglog(xs, ys)
        xf = [min(xs), max(xs)]
        ax.plot(xf, [10 ** (a + b * math.log10(x)) for x in xf], color=line_color,
                ls="--", lw=1.2, alpha=0.6, zorder=2)
        ax.scatter(xs, ys, zorder=3, **style)
        rho = spearman(xs, ys)
        stats_lines.append(f"{label.split(' (')[0]}: ρ={rho:.2f}")
        print(f"  {title:>4} {label:<20} spearman={rho:.3f}  slope={b:.2f}")
    ax.text(0.97, 0.03, "\n".join(stats_lines), transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="0.25", linespacing=1.4)
    xs_all = [r[xkey] for r in rows]
    lo, hi = math.log10(min(xs_all)), math.log10(max(xs_all))
    for r in rows:
        if r["name"] in annotate and r["s1000"] is not None:
            with_fut = annotate[r["name"]] and xkey == "loc"
            note = f"{r['name']} ({r['fut']} FUT)" if with_fut else r["name"]
            # Points in the right ~30% of the (log) axis get their label on the
            # left so it cannot clip at the panel edge.
            right = (math.log10(r[xkey]) - lo) / (hi - lo) > 0.7
            ax.annotate(note, (r[xkey], r["s1000"]), textcoords="offset points",
                        xytext=(-6, 4) if right else (5, 5),
                        ha="right" if right else "left", fontsize=7.5, color="0.25")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.set_axisbelow(True)


def main():
    rows = load_rows()
    print(f"parsed {len(rows)} libraries")

    fig, (axl, axf) = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
    panel(axl, rows, "loc", "Lines of code (log)",
          "vs. lines of code", ANNOTATE_LOC)
    panel(axf, rows, "fut", "Exported functions under test (log)",
          "vs. functions under test", ANNOTATE_FUT)
    axl.set_ylabel("Total tokens, input + output (log)")

    handles = [
        Line2D([], [], marker="o", color="none", markerfacecolor="none",
               markeredgecolor=C_TURBO_EDGE, markeredgewidth=1.3, markersize=7,
               label="GPT-Turbo (est. upper bound)"),
        Line2D([], [], marker="o", color="none", markerfacecolor=C_500,
               markeredgecolor="black", markersize=7, label="GPT-5.4 · 500 tok"),
        Line2D([], [], marker="D", color="none", markerfacecolor=C_1000,
               markeredgecolor="black", markersize=6.5, label="GPT-5.4 · 1000 tok"),
        Line2D([], [], color="0.35", ls="--", lw=1.2, label="power-law fit"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("TestPilot token usage vs. project size", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    for ext in ("pdf", "png"):
        out = os.path.join(FIG, f"gpt5_token_scaling_scatter.{ext}")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"wrote {os.path.relpath(out, HERE)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
