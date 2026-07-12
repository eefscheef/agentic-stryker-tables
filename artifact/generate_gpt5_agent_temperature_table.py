#!/usr/bin/env python3
"""
Generate the compact GPT-5.4 temperature-sensitivity table.

Compares the single temperature-0 full+residual run against the two temperature-1
full+residual HEAD replicates (R3/R4) already used by the residual table.

The temperature-0 run is read from
data/gpt-5.4-agent-residual/head-63be77f/residual-full-gpt54-temp0-20260530/
(vendored from agentic-stryker/runs/temp/, see its manifest.json); set the
TEMP0_RUN_DIR environment variable to point at a different copy.

Outputs:
  tables/tab-gpt5-agent-residual-scores.tex   — residual effect at both
      temperatures (this script owns that table, NOT the residual generator)
  tables/tab-gpt5-agent-temperature.csv       — feeds the temperature dumbbell
      figure (generate_gpt5_temperature_figure.py)
"""

import csv
import json
import math
import os
import statistics
import sys

sys.dont_write_bytecode = True

import generate_gpt5_agent_residual_table as residual


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR = os.path.join(SCRIPT_DIR, "tables")

TEMP0_RUN_DEFAULT = os.path.join(
    SCRIPT_DIR,
    "data",
    "gpt-5.4-agent-residual",
    "head-63be77f",
    "residual-full-gpt54-temp0-20260530",
)

WARN = "%WARNING: auto-generated. Edit generate_gpt5_agent_temperature_table.py instead.\n"
EXPECTED_AGENT_SHA_PREFIX = "63be77f"
INCOMPATIBLE_LIBS = {"bluebird", "js-sdsl", "memfs", "rsvp"}
EPS = 1e-9


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def resolve_temp0_run_dir():
    override = os.environ.get("TEMP0_RUN_DIR")
    if override:
        path = os.path.abspath(os.path.expanduser(override))
        if not os.path.isdir(path):
            sys.exit(f"TEMP0_RUN_DIR does not exist or is not a directory: {path}")
        return path

    if os.path.isdir(TEMP0_RUN_DEFAULT):
        return TEMP0_RUN_DEFAULT
    sys.exit(
        "missing temp-0 run directory; set TEMP0_RUN_DIR or restore "
        f"{TEMP0_RUN_DEFAULT}"
    )


def empty_cell(status):
    return {
        "status": status,
        "duration_ms": None,
        "input_tokens": None,
        "cached_input_tokens": None,
        "output_tokens": None,
        "reasoning_tokens": None,
        "requests": None,
        "tests_written": None,
        "tests_passing": None,
        "tests_failures": None,
        "tests_total": None,
        "fatal_error": None,
        "score_measurement_complete": False,
        "score_splices": [],
        "pre_score": None,
        "residual_score": None,
        "total_valid": None,
        "benchmark_score": None,
        "benchmark_total_valid": None,
        "benchmark_undetected": None,
        "benchmark_exit_code": None,
        "benchmark_delta": None,
        "benchmark_minus_agent_score": None,
    }


def collect_run_from_dir(run_dir):
    """Return {lib: per-cell dict} using the residual table scoring convention."""
    suite = read_json(os.path.join(run_dir, "suite-report.json")) or {}
    suite_status = {
        entry.get("libraryId"): entry.get("status")
        for entry in suite.get("libraries", [])
    }

    out = {}
    for lib in residual.DOMAIN_MAP:
        report = read_json(os.path.join(run_dir, lib, "report", "report.json"))
        if report is None:
            if lib in suite_status:
                out[lib] = residual.finalize_score_cell(
                    empty_cell(suite_status.get(lib))
                )
            continue

        ar = report.get("agentRun", {})
        gen = ar.get("generation", {})
        llm = ar.get("llm", {})
        stats = report.get("stats", {})
        pre = gen.get("preResidualAggregatePass") or {}
        agg = gen.get("aggregatePass") or {}
        stryker = report.get("stryker") or {}
        stryker_metrics = stryker.get("metrics") or {}

        pre_score = pre.get("mutationScore") if pre.get("succeeded") else None
        residual_score = (
            agg.get("mutationScore") if agg.get("succeeded") else None
        )
        benchmark_score = (
            stryker_metrics.get("mutationScore") if stryker_metrics else None
        )

        cell = {
                "status": suite_status.get(lib),
                "duration_ms": ar.get("durationMs"),
                "input_tokens": llm.get("inputTokens"),
                "cached_input_tokens": llm.get("cacheReadInputTokens"),
                "output_tokens": llm.get("outputTokens"),
                "reasoning_tokens": llm.get("reasoningTokens", 0),
                "requests": llm.get("requests"),
                "tests_written": gen.get("testsWritten"),
                "tests_passing": stats.get("nrPasses"),
                "tests_failures": stats.get("nrFailures"),
                "tests_total": stats.get("nrTests"),
                "fatal_error": ar.get("fatalError"),
                "pre_score": pre_score,
                "residual_score": residual_score,
                "total_valid": (
                    agg.get("totalValid")
                    if agg.get("succeeded")
                    else pre.get("totalValid")
                    if pre.get("succeeded")
                    else None
                ),
                "benchmark_score": benchmark_score,
                "benchmark_total_valid": (
                    stryker_metrics.get("totalValid") if stryker_metrics else None
                ),
                "benchmark_undetected": (
                    stryker_metrics.get("totalUndetected")
                    if stryker_metrics
                    else None
                ),
                "benchmark_exit_code": stryker.get("exitCode"),
                "benchmark_delta": (
                    benchmark_score - pre_score
                    if benchmark_score is not None and pre_score is not None
                    else None
                ),
                "benchmark_minus_agent_score": (
                    benchmark_score - residual_score
                    if benchmark_score is not None
                    and residual_score is not None
                    else None
                ),
        }
        out[lib] = residual.finalize_score_cell(cell)
    return out, suite


def mean(values):
    clean = [v for v in values if v is not None]
    return statistics.fmean(clean) if clean else None


def median(values):
    clean = [v for v in values if v is not None]
    return statistics.median(clean) if clean else None


def wilcoxon(diffs):
    """Two-sided Wilcoxon signed-rank normal approximation with tie correction."""
    nonzero = [d for d in diffs if d is not None and abs(d) > EPS]
    n = len(nonzero)
    if n == 0:
        return {"n": 0, "W": None, "z": None, "p": None}

    sorted_diffs = sorted(
        [{"abs": abs(d), "sign": 1 if d > 0 else -1} for d in nonzero],
        key=lambda item: item["abs"],
    )
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(sorted_diffs[j + 1]["abs"] - sorted_diffs[i]["abs"]) <= EPS:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[k] = rank
        i = j + 1

    w_pos = 0.0
    w_neg = 0.0
    ties = {}
    for item, rank in zip(sorted_diffs, ranks):
        if item["sign"] > 0:
            w_pos += rank
        else:
            w_neg += rank
        key = round(item["abs"], 10)
        ties[key] = ties.get(key, 0) + 1

    w_stat = min(w_pos, w_neg)
    mu = n * (n + 1) / 4.0
    tie_correction = sum(count ** 3 - count for count in ties.values())
    sigma = math.sqrt(
        n * (n + 1) * (2 * n + 1) / 24.0 - tie_correction / 48.0
    )
    z = 0.0
    if sigma > 0:
        z = (w_stat - mu + 0.5 * math.copysign(1, mu - w_stat)) / sigma
    p = 2 * (1 - (0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))))
    return {
        "n": n,
        "W": w_stat,
        "z": z,
        "p": p,
        "median": median(nonzero),
        "pos": sum(1 for d in nonzero if d > 0),
        "neg": sum(1 for d in nonzero if d < 0),
    }


def delta_for_cell(cell):
    if cell is None:
        return None
    if cell.get("pre_score") is None or cell.get("residual_score") is None:
        return None
    return cell["residual_score"] - cell["pre_score"]


def display_status(cell):
    if cell is None:
        return None
    return cell.get("status")


def display_score_status(cell):
    status = residual.display_status(cell)
    return None if status == "--" else status


def build_rows(temp0_run, temp1_runs, temp1_rows):
    rows = []
    for lib in residual.DOMAIN_MAP:
        t0 = temp0_run.get(lib)
        r3 = temp1_runs["3"].get(lib)
        r4 = temp1_runs["4"].get(lib)
        pre_t1, pre_t1_n = residual.avg([
            r3.get("pre_score") if r3 else None,
            r4.get("pre_score") if r4 else None,
        ])
        final_t1, final_t1_n = residual.avg([
            r3.get("residual_score") if r3 else None,
            r4.get("residual_score") if r4 else None,
        ])
        delta_t1 = temp1_rows[lib]["delta"]
        delta_t1_n = temp1_rows[lib]["delta_n"]
        pre_t0 = t0.get("pre_score") if t0 else None
        final_t0 = t0.get("residual_score") if t0 else None
        delta_t0 = delta_for_cell(t0)

        rows.append({
            "library": lib,
            "status_temp0": display_status(t0),
            "pre_score_temp0": pre_t0,
            "final_score_temp0": final_t0,
            "delta_residual_temp0": delta_t0,
            "status_r3": display_status(r3),
            "status_r4": display_status(r4),
            "display_status_temp0": display_score_status(t0),
            "display_status_temp1": "/".join([
                display_score_status(r3) or "--",
                display_score_status(r4) or "--",
            ]),
            "pre_score_temp1_mean": pre_t1,
            "pre_score_temp1_n": pre_t1_n,
            "final_score_temp1_mean": final_t1,
            "final_score_temp1_n": final_t1_n,
            "delta_residual_temp1_mean": delta_t1,
            "delta_residual_temp1_n": delta_t1_n,
            "delta_pre_temp0_minus_temp1": (
                pre_t0 - pre_t1
                if pre_t0 is not None and pre_t1 is not None
                else None
            ),
            "delta_final_temp0_minus_temp1": (
                final_t0 - final_t1
                if final_t0 is not None and final_t1 is not None
                else None
            ),
            "delta_residual_temp0_minus_temp1": (
                delta_t0 - delta_t1
                if delta_t0 is not None and delta_t1 is not None
                else None
            ),
        })
    return rows


def direction_counts(values):
    clean = [v for v in values if v is not None]
    pos = sum(1 for v in clean if v > EPS)
    neg = sum(1 for v in clean if v < -EPS)
    zero = len(clean) - pos - neg
    return {"pos": pos, "neg": neg, "zero": zero, "n": len(clean)}


def summarize(rows, temp0_run, temp1_runs, temp0_suite, temp0_run_dir):
    t0_pre = [row["pre_score_temp0"] for row in rows]
    t1_pre = [row["pre_score_temp1_mean"] for row in rows]
    d_pre = [row["delta_pre_temp0_minus_temp1"] for row in rows]
    t0_final = [row["final_score_temp0"] for row in rows]
    t1_final = [row["final_score_temp1_mean"] for row in rows]
    d_final = [row["delta_final_temp0_minus_temp1"] for row in rows]
    t0_residual = [row["delta_residual_temp0"] for row in rows]
    t1_residual = [row["delta_residual_temp1_mean"] for row in rows]
    d_residual = [row["delta_residual_temp0_minus_temp1"] for row in rows]

    temp0_no_test_libs = [
        lib for lib in residual.DOMAIN_MAP
        if (temp0_run.get(lib) or {}).get("no_test_score")
    ]
    temp1_no_test_libs = sorted({
        lib
        for lib in residual.DOMAIN_MAP
        for run_id in residual.RUN_IDS
        if (temp1_runs[run_id].get(lib) or {}).get("no_test_score")
    })
    temp1_no_test_run_count = sum(
        1
        for lib in residual.DOMAIN_MAP
        for run_id in residual.RUN_IDS
        if (temp1_runs[run_id].get(lib) or {}).get("no_test_score")
    )

    status_by_lib = {
        entry.get("libraryId"): entry.get("status")
        for entry in temp0_suite.get("libraries", [])
    }
    temp0_statuses = [status_by_lib.get(lib) for lib in residual.DOMAIN_MAP]
    temp0_partial = sum(1 for status in temp0_statuses if (status or "").startswith("partial"))
    temp0_failed = sum(1 for status in temp0_statuses if (status or "").startswith("failed"))
    temp0_completed = sum(1 for status in temp0_statuses if status == "completed")
    temp0_scored = sum(
        1
        for lib in residual.DOMAIN_MAP
        if (temp0_run.get(lib) or {}).get("pre_score") is not None
        and (temp0_run.get(lib) or {}).get("residual_score") is not None
    )
    temp1_scored = sum(
        1
        for row in rows
        if row["pre_score_temp1_mean"] is not None
        and row["final_score_temp1_mean"] is not None
    )

    summary = {
        "temp0_run_dir": temp0_run_dir,
        "temp0_completed": temp0_completed,
        "temp0_partial": temp0_partial,
        "temp0_failed": temp0_failed,
        "temp0_scored": temp0_scored,
        "temp1_scored": temp1_scored,
        "temp0_no_test_libs": temp0_no_test_libs,
        "temp1_no_test_libs": temp1_no_test_libs,
        "temp1_no_test_run_count": temp1_no_test_run_count,
        "pre_temp0_mean": mean(t0_pre),
        "pre_temp1_mean": mean(t1_pre),
        "pre_delta_mean": mean(d_pre),
        "pre_temp0_median": median(t0_pre),
        "pre_temp1_median": median(t1_pre),
        "pre_delta_median": median(d_pre),
        "pre_wilcoxon": wilcoxon(d_pre),
        "final_temp0_mean": mean(t0_final),
        "final_temp1_mean": mean(t1_final),
        "final_delta_mean": mean(d_final),
        "final_temp0_median": median(t0_final),
        "final_temp1_median": median(t1_final),
        "final_delta_median": median(d_final),
        "final_wilcoxon": wilcoxon(d_final),
        "residual_temp0_mean": mean(t0_residual),
        "residual_temp1_mean": mean(t1_residual),
        "residual_delta_mean": mean(d_residual),
        "residual_temp0_median": median(t0_residual),
        "residual_temp1_median": median(t1_residual),
        "residual_delta_median": median(d_residual),
        "residual_temp0_directions": direction_counts(t0_residual),
        "residual_temp1_directions": direction_counts(t1_residual),
        "residual_temp0_wilcoxon": wilcoxon(t0_residual),
        "incompatible_skipped": sorted([
            entry.get("libraryId")
            for entry in temp0_suite.get("libraries", [])
            if entry.get("status") == "skipped"
            and entry.get("libraryId") in INCOMPATIBLE_LIBS
        ]),
    }
    return summary


def validate(summary, temp0_suite, temp0_run):
    """Golden regression guard against the shipped data snapshot.

    The hardcoded expectations below match the vendored temp-0 / R3 / R4
    data exactly and sys.exit on any mismatch. When regenerating data,
    update these expected values deliberately -- do not loosen them.
    """
    problems = []

    def expect_equal(label, actual, expected):
        if actual != expected:
            problems.append(f"{label}: expected {expected!r}, got {actual!r}")

    def expect_close(label, actual, expected, tolerance):
        if actual is None or abs(actual - expected) > tolerance:
            problems.append(
                f"{label}: expected about {expected}, got {actual}"
            )

    expect_equal("temp0 suite model", temp0_suite.get("model"), "gpt-5.4")
    expect_equal("temp0 suite temperature", temp0_suite.get("temperature"), 0)
    expect_equal("temp0 mutation feedback", temp0_suite.get("mutationFeedback"), "full")
    expect_equal("temp0 dry-run repair", temp0_suite.get("strykerDryRunRepair"), "on")
    expect_equal("temp0 residual feedback", temp0_suite.get("residualMutationFeedback"), True)
    expect_equal("temp0 residual max items", temp0_suite.get("residualMutationMaxItems"), 5)
    expect_equal("temp0 completed libraries", summary["temp0_completed"], 21)
    expect_equal("temp0 partial libraries", summary["temp0_partial"], 0)
    expect_equal("temp0 failed libraries", summary["temp0_failed"], 0)
    expect_equal("temp0 scored libraries", summary["temp0_scored"], 21)
    expect_equal("temp0 skipped incompatible libraries", summary["incompatible_skipped"], sorted(INCOMPATIBLE_LIBS))

    for lib in residual.DOMAIN_MAP:
        report = read_json(os.path.join(summary["temp0_run_dir"], lib, "report", "report.json"))
        if report is None:
            problems.append(f"temp0 {lib}: missing report.json")
            continue
        ar = report.get("agentRun", {})
        sha = ar.get("agentGitSha") or ""
        if not sha.startswith(EXPECTED_AGENT_SHA_PREFIX):
            problems.append(f"temp0 {lib}: expected agent sha {EXPECTED_AGENT_SHA_PREFIX}, got {sha!r}")
        expect_equal(f"temp0 {lib} model", ar.get("model"), "gpt-5.4")
        expect_equal(f"temp0 {lib} mutation feedback", ar.get("mutationFeedback"), "full")
        expect_equal(f"temp0 {lib} dry-run repair", ar.get("strykerDryRunRepair"), "on")

    expect_close("pre-residual paired delta mean", summary["pre_delta_mean"], 1.8, 0.1)
    expect_close("pre-residual paired delta median", summary["pre_delta_median"], -0.1, 0.1)
    expect_close("pre-residual Wilcoxon p", summary["pre_wilcoxon"]["p"], 0.54, 0.02)
    expect_close("final paired delta mean", summary["final_delta_mean"], 3.1, 0.1)
    expect_close("final paired delta median", summary["final_delta_median"], 0.0, 0.1)
    expect_close("final Wilcoxon p", summary["final_wilcoxon"]["p"], 0.88, 0.02)
    expect_close("temp0 residual mean", summary["residual_temp0_mean"], 9.3, 0.1)
    expect_close("temp0 residual median", summary["residual_temp0_median"], 5.6, 0.1)
    expect_equal("temp0 residual positive nonzero directions", summary["residual_temp0_directions"]["pos"], 17)
    expect_equal("temp0 residual negative nonzero directions", summary["residual_temp0_directions"]["neg"], 1)
    expect_equal("temp0 residual nonzero count", summary["residual_temp0_wilcoxon"]["n"], 18)
    expect_close("temp0 residual Wilcoxon p", summary["residual_temp0_wilcoxon"]["p"], 0.0006, 0.001)
    expect_close("temp1 residual mean", summary["residual_temp1_mean"], 8.0, 0.1)
    expect_close("temp1 residual median", summary["residual_temp1_median"], 3.9, 0.1)
    expect_equal("temp1 residual positive nonzero directions", summary["residual_temp1_directions"]["pos"], 16)
    expect_equal("temp1 residual negative nonzero directions", summary["residual_temp1_directions"]["neg"], 1)

    expect_equal("temp1 no-test libraries", summary["temp1_no_test_libs"], ["crawler-url-parser", "uneval"])
    expect_equal("temp0 no-test libraries", summary["temp0_no_test_libs"], [])
    for lib in ["crawler-url-parser", "uneval"]:
        cell = temp0_run.get(lib) or {}
        if not cell.get("score_measurement_complete"):
            problems.append(f"temp0 {lib}: expected complete score measurement")

    if problems:
        sys.exit(
            "validation failed; refusing to write thesis-facing temperature table:\n"
            + "\n".join(f"- {problem}" for problem in problems)
        )
    return []


def fmt_pct(value, star=False):
    if value is None:
        return "--"
    suffix = residual.STAR if star else ""
    return f"{value:.1f}\\%{suffix}"


def fmt_delta(value, star=False):
    if value is None:
        return "--"
    sign = "+" if value >= 0 else "-"
    suffix = residual.STAR if star else ""
    return f"${sign}${abs(value):.1f}{suffix}"


def temp1_star(row, key):
    return 0 < row[key] < len(residual.RUN_IDS)


def write_csv(rows, path):
    fields = [
        "library",
        "status_temp0",
        "pre_score_temp0",
        "final_score_temp0",
        "delta_residual_temp0",
        "status_r3",
        "status_r4",
        "pre_score_temp1_mean",
        "pre_score_temp1_n",
        "final_score_temp1_mean",
        "final_score_temp1_n",
        "delta_residual_temp1_mean",
        "delta_residual_temp1_n",
        "delta_pre_temp0_minus_temp1",
        "delta_final_temp0_minus_temp1",
        "delta_residual_temp0_minus_temp1",
    ]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def print_summary(summary, table_path, csv_path, warnings=None):
    pre_w = summary["pre_wilcoxon"]
    final_w = summary["final_wilcoxon"]
    residual_w = summary["residual_temp0_wilcoxon"]
    t0_dirs = summary["residual_temp0_directions"]
    t1_dirs = summary["residual_temp1_directions"]

    print(f"wrote {os.path.relpath(table_path, SCRIPT_DIR)}")
    print(f"wrote {os.path.relpath(csv_path, SCRIPT_DIR)}")
    if warnings:
        print("aggregate validation warnings:")
        for warning in warnings:
            print(f"  - {warning}")
        print()
    print()
    print(f"temperature-0 run dir: {summary['temp0_run_dir']}")
    print(
        "temperature-0 provenance: "
        f"agent commit {EXPECTED_AGENT_SHA_PREFIX}, model gpt-5.4, "
        "temperature 0, full feedback, residual max-items=5, "
        "strykerDryRunRepair=on"
    )
    print(
        "temperature-0 status: "
        f"{summary['temp0_completed']} completed/scored, "
        f"{summary['temp0_partial']} partial, {summary['temp0_failed']} failed"
    )
    print(
        "no-test libraries: "
        f"Temp. 0={len(summary['temp0_no_test_libs'])}; "
        f"Temp. 1={len(summary['temp1_no_test_libs'])} "
        f"({', '.join(summary['temp1_no_test_libs'])}; "
        f"{summary['temp1_no_test_run_count']} run-level outcomes)"
    )
    print()
    print(
        "pre-residual score: "
        f"Temp. 0 mean={summary['pre_temp0_mean']:.2f}% "
        f"median={summary['pre_temp0_median']:.2f}%; "
        f"Temp. 1 mean={summary['pre_temp1_mean']:.2f}% "
        f"median={summary['pre_temp1_median']:.2f}%; "
        f"paired delta mean={summary['pre_delta_mean']:+.2f}pp "
        f"median={summary['pre_delta_median']:+.2f}pp "
        f"Wilcoxon p={pre_w['p']:.4f} (n={pre_w['n']})"
    )
    print(
        "agent-final score: "
        f"Temp. 0 mean={summary['final_temp0_mean']:.2f}% "
        f"median={summary['final_temp0_median']:.2f}%; "
        f"Temp. 1 mean={summary['final_temp1_mean']:.2f}% "
        f"median={summary['final_temp1_median']:.2f}%; "
        f"paired delta mean={summary['final_delta_mean']:+.2f}pp "
        f"median={summary['final_delta_median']:+.2f}pp "
        f"Wilcoxon p={final_w['p']:.4f} (n={final_w['n']})"
    )
    print(
        "residual effect: "
        f"Temp. 0 mean={summary['residual_temp0_mean']:+.2f}pp "
        f"median={summary['residual_temp0_median']:+.2f}pp "
        f"directions={t0_dirs['pos']}+/{t0_dirs['neg']}-/{t0_dirs['zero']} zero "
        f"Wilcoxon p={residual_w['p']:.4f} (n={residual_w['n']}); "
        f"Temp. 1 mean={summary['residual_temp1_mean']:+.2f}pp "
        f"median={summary['residual_temp1_median']:+.2f}pp "
        f"directions={t1_dirs['pos']}+/{t1_dirs['neg']}-/{t1_dirs['zero']} zero; "
        f"T0-T1 mean={summary['residual_delta_mean']:+.2f}pp "
        f"median={summary['residual_delta_median']:+.2f}pp"
    )


def write_residual_temp_table(rows, summary, path):
    """Grouped residual-effect table at both decoding temperatures.

    Replaces the residual generator's temperature-1-only scores table: it adds
    the temperature-0 columns so the residual effect (pre -> final) can be read
    at both temperatures. Aggregates exclude EXCLUDE_LIBS (fs-extra), matching
    the rest of the thesis. Written to tab-gpt5-agent-residual-scores.tex with
    the existing label, so the residual subsection picks it up unchanged.
    """
    incl = [r for r in rows if r["library"] not in residual.EXCLUDE_LIBS]
    _m = lambda key: mean([r.get(key) for r in incl])
    _md = lambda key: median([r.get(key) for r in incl])
    lines = [
        WARN.rstrip("\n"),
        r"\begin{table*}[!htb]",
        r"\centering",
        r"\caption{Residual feedback effect at both decoding temperatures. "
        r"\emph{Pre-residual} and \emph{Agent final} are the agent-side "
        r"\texttt{strykerMutationTest} scores before and after the residual phase, "
        r"and $\Delta$ is their difference (Agent final $-$ Pre-residual). "
        r"Temperature 1 is the mean of two runs and temperature 0 "
        r"is a single run. No-test failures are scored as 0.0\%. "
        r"Cells marked \textsuperscript{*} were taken "
        r"from a single run.}",
        r"\label{tab:gpt5-agent-residual-scores}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l r r r r r r}",
        r"\toprule",
        r"& \multicolumn{3}{c}{\textbf{Temperature 1 (mean of two runs)}} & "
        r"\multicolumn{3}{c}{\textbf{Temperature 0}} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"\textbf{Project} & \textbf{Pre-residual} & \textbf{Agent final} & "
        r"\textbf{$\Delta$} & \textbf{Pre-residual} & \textbf{Agent final} & "
        r"\textbf{$\Delta$} \\",
        r"\midrule",
    ]
    for row in rows:
        lib = row["library"]
        if lib in residual.EXCLUDE_LIBS:
            continue
        cells = [
            fmt_pct(row.get("pre_score_temp1_mean"), temp1_star(row, "pre_score_temp1_n")),
            fmt_pct(row.get("final_score_temp1_mean"), temp1_star(row, "final_score_temp1_n")),
            fmt_delta(row.get("delta_residual_temp1_mean"), temp1_star(row, "delta_residual_temp1_n")),
            fmt_pct(row.get("pre_score_temp0")),
            fmt_pct(row.get("final_score_temp0")),
            fmt_delta(row.get("delta_residual_temp0")),
        ]
        lines.append(f"{lib} & " + " & ".join(cells) + r" \\")
        if lib in residual.SEPARATOR_LIBS:
            lines.append(r"\midrule")
    lines.append(r"\midrule")
    lines.append(
        r"\textbf{Mean} & "
        + fmt_pct(_m("pre_score_temp1_mean")) + " & "
        + fmt_pct(_m("final_score_temp1_mean")) + " & "
        + fmt_delta(_m("delta_residual_temp1_mean")) + " & "
        + fmt_pct(_m("pre_score_temp0")) + " & "
        + fmt_pct(_m("final_score_temp0")) + " & "
        + fmt_delta(_m("delta_residual_temp0")) + r" \\"
    )
    lines.append(
        r"\textbf{Median} & "
        + fmt_pct(_md("pre_score_temp1_mean")) + " & "
        + fmt_pct(_md("final_score_temp1_mean")) + " & "
        + fmt_delta(_md("delta_residual_temp1_mean")) + " & "
        + fmt_pct(_md("pre_score_temp0")) + " & "
        + fmt_pct(_md("final_score_temp0")) + " & "
        + fmt_delta(_md("delta_residual_temp0")) + r" \\"
    )
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def main():
    temp0_run_dir = resolve_temp0_run_dir()
    temp0_run, temp0_suite = collect_run_from_dir(temp0_run_dir)
    rates = residual.load_pricing()
    temp1_runs = {run_id: residual.collect_run(run_id) for run_id in residual.RUN_IDS}
    temp1_rows = residual.aggregate(temp1_runs, rates)
    rows = build_rows(temp0_run, temp1_runs, temp1_rows)
    summary = summarize(rows, temp0_run, temp1_runs, temp0_suite, temp0_run_dir)

    warnings = validate(summary, temp0_suite, temp0_run)

    os.makedirs(TABLES_DIR, exist_ok=True)
    # This generator now owns the residual scores table (with temperature-0
    # columns added); the temperature-1-only version is no longer written by
    # generate_gpt5_agent_residual_table.py. The temperature CSV still feeds the
    # temperature dumbbell (generate_gpt5_temperature_figure.py).
    table_path = os.path.join(TABLES_DIR, "tab-gpt5-agent-residual-scores.tex")
    csv_path = os.path.join(TABLES_DIR, "tab-gpt5-agent-temperature.csv")
    write_residual_temp_table(rows, summary, table_path)
    write_csv(rows, csv_path)
    print_summary(summary, table_path, csv_path, warnings)


if __name__ == "__main__":
    main()
