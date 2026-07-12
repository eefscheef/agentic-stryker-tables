#!/usr/bin/env python3
"""Generate tab-nontriviality-testpilot-vs-agent.tex (label tab:nontriviality).

Non-triviality of generated tests, comparing GPT-Turbo + GPT-5.4 TestPilot
against the agent runs. One row per generator/condition:

  Runs   number of runs averaged
  Tests  per-run test count (mean over runs, rounded half-up)
  Pass   share of generated tests that pass
  NT*    share of generated tests that contain a non-trivial assertion

Aggregation (matches the original hand-built table):
  * Per run, Pass and NT* are aggregate ratios over that run's packages.
  * Multi-run rows report the MEAN of the per-run ratios (not the pooled ratio),
    and Tests is the mean of the per-run test counts.

Sources:
  GPT-Turbo:  data/gptturbo/<run>/<pkg>/report.json  (stats; 10 runs)
              original package-name / `assert` non-triviality criterion.
  TestPilot:  data/gpt-5.4-output-{500,1000}/1/<pkg>/report.json (stats; 1 run each)
              same original criterion (report.json nrNonTrivialTests).
  Agent:      tables/tab-gpt5-agent-nontriviality.csv (CodeQL pipeline, produced by
              generate_gpt5_agent_nontriviality.py). Agent-compatible criterion
              (relative-path + node:assert* imports); the strict criterion is 0%
              for every agent run. Rows group the CSV run_ids:
                full+residual T=1 = mean(R3, R4)
                hidden       T=1 = mean(H1, H2)
                full+residual T=0 = T0

The TestPilot rows are data-driven, so re-running this after the 2026-06-27
fixed-harness crawler-url-parser splice automatically picks up the new numbers.

Output: tables/tab-nontriviality-testpilot-vs-agent.tex
"""
import csv
import glob
import json
import os
import statistics
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
TABLES_DIR = os.path.join(SCRIPT_DIR, "tables")

GPTTURBO_DIR = os.path.join(DATA_DIR, "gptturbo")
GPT54_500_RUN = os.path.join(DATA_DIR, "gpt-5.4-output-500", "1")
GPT54_1000_RUN = os.path.join(DATA_DIR, "gpt-5.4-output-1000", "1")
AGENT_CSV = os.path.join(TABLES_DIR, "tab-gpt5-agent-nontriviality.csv")
OUTPUT_TEX = os.path.join(TABLES_DIR, "tab-nontriviality-testpilot-vs-agent.tex")

# Excluded thesis-wide for now (dropped from every aggregate row: GPT-Turbo,
# TestPilot, and agent).
EXCLUDE_LIBS = {"fs-extra"}

# Agent CSV run_ids grouped into table rows, in display order.
AGENT_GROUPS = [
    (r"Agent, full+residual ($T{=}1$)", ["R3", "R4"]),
    (r"Agent, hidden ($T{=}1$)", ["H1", "H2"]),
    (r"Agent, full+residual ($T{=}0$)", ["T0"]),
]

# Golden values guard against silent drift. fs-extra is now excluded thesis-wide
# from every aggregate row: it carried many overfit tests (TestPilot Tests drop
# ~1.3k at 500 / ~1.5k at 1000), and removing it lifts NT* ~0.6-1.1 pp because its
# tests were below-average non-trivial. The fixed-harness crawler-url-parser splice
# (2026-06-27) is already reflected in the TestPilot rows.
# Tuples are (runs, tests, pass%, NT*-all%, NT*-passing%). NT*-passing is the share
# of passing tests that are non-trivial: for GPT-Turbo it is lower than NT*-all (its
# failing tests skew non-trivial), for GPT-5.4 TestPilot it is higher, and for the
# agent the two coincide because nearly every test passes.
EXPECTED = {
    "GPT-Turbo (original)": (10, 4984, 42.3, 71.2, 67.8),
    r"\textsc{TestPilot}-500": (1, 8872, 34.5, 92.3, 97.2),
    r"\textsc{TestPilot}-1000": (1, 9357, 34.7, 92.7, 97.4),
    r"Agent, full+residual ($T{=}1$)": (2, 827, 99.9, 99.1, 99.1),
    r"Agent, hidden ($T{=}1$)": (2, 821, 99.9, 99.6, 99.6),
    r"Agent, full+residual ($T{=}0$)": (1, 848, 100.0, 98.1, 98.1),
}


def round_half_up(value):
    return int(value + 0.5)


def fmt_int(value):
    """Integer with a LaTeX thousands separator (e.g. 10180 -> 10{,}180)."""
    s = f"{value:,}"
    return s.replace(",", "{,}")


def run_stats_from_reports(run_dir):
    """Aggregate (tests, passes, nontrivial, nontrivial_passes) over every package
    report in a run."""
    tests = passes = nontrivial = nontrivial_passes = 0
    for report in glob.glob(os.path.join(run_dir, "*", "report.json")):
        if os.path.basename(os.path.dirname(report)) in EXCLUDE_LIBS:
            continue
        stats = json.load(open(report)).get("stats", {})
        tests += stats.get("nrTests", 0) or 0
        passes += stats.get("nrPasses", 0) or 0
        nontrivial += stats.get("nrNonTrivialTests", 0) or 0
        nontrivial_passes += stats.get("nrNonTrivialPasses", 0) or 0
    return tests, passes, nontrivial, nontrivial_passes


def row_from_runs(label, per_run):
    """Build a [label, runs, tests, pass%, nt%, nt-pass%] row from per-run
    (tests, passes, nontrivial, nontrivial_passes) tuples, using
    mean-of-per-run-ratios for the percentages. nt-pass% is the share of
    *passing* tests that are non-trivial (vs nt% over all generated tests)."""
    if not per_run or any(t == 0 for t, *_ in per_run):
        sys.exit(f"{label}: a run has zero tests; cannot aggregate")
    tests = round_half_up(statistics.mean(t for t, *_ in per_run))
    pass_pct = 100 * statistics.mean(p / t for t, p, _, _ in per_run)
    nt_pct = 100 * statistics.mean(nt / t for t, _, nt, _ in per_run)
    nt_pass_pct = 100 * statistics.mean((ntp / p if p else 0.0) for t, p, _, ntp in per_run)
    return [label, len(per_run), tests, round(pass_pct, 1),
            round(nt_pct, 1), round(nt_pass_pct, 1)]


def collect_gptturbo_row():
    runs = sorted(d for d in glob.glob(os.path.join(GPTTURBO_DIR, "*")) if os.path.isdir(d))
    per_run = [run_stats_from_reports(run) for run in runs]
    return row_from_runs("GPT-Turbo (original)", per_run)


def collect_testpilot_row(label, run_dir):
    return row_from_runs(label, [run_stats_from_reports(run_dir)])


def collect_agent_rows():
    rows = list(csv.DictReader(open(AGENT_CSV)))
    by_run = {}
    for r in rows:
        by_run.setdefault(r["run_id"], []).append(r)

    def run_triple(run_id):
        recs = [x for x in by_run[run_id] if x["library"] not in EXCLUDE_LIBS]
        tests = sum(int(x["tests_total"]) for x in recs)
        passes = sum(int(x["tests_passing"]) for x in recs)
        nontrivial = sum(int(x["nontrivial_tests_agent"]) for x in recs)
        nontrivial_passes = sum(int(x["nontrivial_passes_agent"]) for x in recs)
        return tests, passes, nontrivial, nontrivial_passes

    out = []
    for label, run_ids in AGENT_GROUPS:
        missing = [r for r in run_ids if r not in by_run]
        if missing:
            sys.exit(f"{label}: missing agent run_ids {missing} in {AGENT_CSV}")
        out.append(row_from_runs(label, [run_triple(r) for r in run_ids]))
    return out


def validate(rows):
    # Golden regression guard: expectations below match the shipped data
    # snapshot and sys.exit on mismatch. Update them deliberately when the
    # data is regenerated.
    errors = []
    for label, runs, tests, pass_pct, nt_pct, nt_pass_pct in rows:
        want = EXPECTED.get(label)
        if want is None:
            errors.append(f"unexpected row: {label}")
            continue
        got = (runs, tests, round(pass_pct, 1), round(nt_pct, 1), round(nt_pass_pct, 1))
        if got != want:
            errors.append(f"{label}: expected {want}, got {got}")
    if errors:
        sys.exit("nontriviality table validation failed:\n" + "\n".join(errors))


def render(rows):
    by_label = {r[0]: r for r in rows}

    def tex_row(label):
        _, runs, tests, pass_pct, nt_pct, nt_pass_pct = by_label[label]
        return (f"{label} & {runs} & {fmt_int(tests)} & "
                f"{pass_pct:.1f}\\% & {nt_pct:.1f}\\% & {nt_pass_pct:.1f}\\% \\\\")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\caption{Non-triviality of generated tests. \emph{Pass} is the share of",
        r"generated tests that pass. \emph{NT}\textsuperscript{*} is the percentage of",
        r"all generated tests that contain a non-trivial assertion, and",
        r"\emph{NT}\textsuperscript{*}~(pass.)\ the same percentage but for passing",
        r"tests only. \textsc{TestPilot} and GPT-Turbo use the original criterion,",
        r"while agent rows use the agent-compatible metric.}",
        r"\label{tab:nontriviality}",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Generator (condition) & Runs & Tests & Pass & NT\textsuperscript{*} & "
        r"NT\textsuperscript{*}~(pass.) \\",
        r"\midrule",
        tex_row("GPT-Turbo (original)"),
        tex_row(r"\textsc{TestPilot}-500"),
        tex_row(r"\textsc{TestPilot}-1000"),
        r"\midrule",
        tex_row(r"Agent, full+residual ($T{=}1$)"),
        tex_row(r"Agent, hidden ($T{=}1$)"),
        tex_row(r"Agent, full+residual ($T{=}0$)"),
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        r"\smallskip",
        r"{\footnotesize\raggedright \textsuperscript{*}Re-scoring the \textsc{TestPilot}/GPT-Turbo suites",
        r"under the agent-compatible criterion does not change their non-triviality score",
        r"($\le$0.02\,pp; zero affected files for GPT-5.4), so the difference does not give",
        r"either system an advantage.\par}",
        r"\end{table}",
        r"",
    ]
    return "\n".join(lines)


def main():
    rows = [collect_gptturbo_row(),
            collect_testpilot_row(r"\textsc{TestPilot}-500", GPT54_500_RUN),
            collect_testpilot_row(r"\textsc{TestPilot}-1000", GPT54_1000_RUN)]
    rows.extend(collect_agent_rows())
    validate(rows)
    with open(OUTPUT_TEX, "w") as fh:
        fh.write(render(rows))
    print(f"wrote {OUTPUT_TEX}")
    for label, runs, tests, pass_pct, nt_pct, nt_pass_pct in rows:
        print(f"  {label:34s} runs={runs} tests={tests:>6} "
              f"pass={pass_pct:5.1f}% nt*={nt_pct:5.1f}% nt*-pass={nt_pass_pct:5.1f}%")


if __name__ == "__main__":
    main()
