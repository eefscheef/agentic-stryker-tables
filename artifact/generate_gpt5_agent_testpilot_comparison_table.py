#!/usr/bin/env python3
"""
Generate a compact aggregate comparison between GPT-5.4 TestPilot and the
final GPT-5.4 agent.

Reads the same source artifacts as the existing TestPilot mutation/token and
agent residual table generators. The agent side uses the HEAD runs (R3/R4) via
generate_gpt5_agent_residual_table. The comparison pairs systems per library.
No-test agent failures follow the residual-table convention and contribute 0.0%,
so the paired comparison now covers all 21 benchmark libraries.

Outputs:
  tables/tab-gpt5-agent-testpilot-comparison.tex
  tables/tab-gpt5-agent-testpilot-comparison.csv
"""

import csv
import json
import os
import statistics
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR = os.path.join(SCRIPT_DIR, "tables")

sys.path.insert(0, SCRIPT_DIR)

import generate_gpt5_agent_residual_table as agent_residual
import generate_gpt5_comparison_tables as comparison
import generate_gpt5_token_table as token_table

SCRIPT_NAME = os.path.basename(__file__)
WARN = f"%WARNING: auto-generated. Edit {SCRIPT_NAME} instead."
OUTPUT_TEX = os.path.join(TABLES_DIR, "tab-gpt5-agent-testpilot-comparison.tex")
OUTPUT_CSV = os.path.join(TABLES_DIR, "tab-gpt5-agent-testpilot-comparison.csv")
OUTPUT_PERLIB_TEX = os.path.join(TABLES_DIR, "tab-gpt5-agent-testpilot-per-library.tex")
OUTPUT_PERLIB_CSV = os.path.join(TABLES_DIR, "tab-gpt5-agent-testpilot-per-library.csv")

_ALL_COMPARISON_LIBS = list(comparison.DOMAIN_MAP)


def _agent_scored_libs():
    """Libraries the agent produced a residual score for under the HEAD runs.

    No-test failures are scored as 0.0% by the residual table, so the current
    data retains all 21 benchmark libraries. Computed from the data, not hardcoded.
    """
    rates = agent_residual.load_pricing()
    runs = {r: agent_residual.collect_run(r) for r in agent_residual.RUN_IDS}
    rows = agent_residual.aggregate(runs, rates)
    return [lib for lib in _ALL_COMPARISON_LIBS if rows[lib]["residual_score"] is not None]


# Every library the agent scored (21). Used by the §2 agent scatter, which plots the
# agent's own coverage vs. mutation per library and must keep fs-extra.
AGENT_SCORED_LIBS = _agent_scored_libs()

# Cross-system comparison set (20). fs-extra runs fine but its TestPilot tests
# env-overfit (a validation artifact whose mutation/coverage do not reflect genuine
# test quality), so it is excluded from the SCORE and COVERAGE aggregates of this
# head-to-head comparison for soundness/consistency with the GPT-Turbo comparison.
COMPARISON_LIBS = [lib for lib in AGENT_SCORED_LIBS
                   if lib not in comparison.OVERFIT_EXCLUDED_LIBS]

# Resource set for the whole-suite descriptors (passing tests, LoC, tokens, cost,
# time). fs-extra is now dropped here too, so these totals are over the same 20
# libraries as the score/coverage columns and match the 20-lib token table.
RESOURCE_LIBS = [lib for lib in comparison.DOMAIN_MAP
                 if lib not in comparison.OVERFIT_EXCLUDED_LIBS]


def mean(values):
    return sum(values) / len(values)


def pct(value):
    return f"{value:.1f}\\%"


def fmt_count(value):
    if value is None:
        return "--"
    return f"{round(value):,}"


# Short display labels for the rows (the long internal keys stay for validation).
DISPLAY_NAME = {
    "GPT-5.4 TestPilot-500": "TestPilot-500",
    "GPT-5.4 TestPilot-1000": "TestPilot-1000",
    "Final agent": "Agent",
}


def fmt_loc(value):
    """Non-blank lines of passing test code, as a compact 'k' figure (109.4k)."""
    if value is None:
        return "--"
    return f"{value / 1000:.1f}k"


def _nonblank_loc(path):
    try:
        with open(path, errors="replace") as fh:
            return sum(1 for ln in fh.read().splitlines() if ln.strip())
    except FileNotFoundError:
        return 0


def fmt_testpilot_token(value):
    """Use fixed precision so token totals compare cleanly across systems."""
    if value is None:
        return "--"
    rounded = round(value / 1000) * 1000
    if rounded >= 1_000_000:
        return f"{rounded / 1_000_000:.2f}M"
    if rounded >= 1_000:
        return f"{rounded // 1000}k"
    return str(rounded)


def fmt_agent_token(value):
    """Match generate_gpt5_agent_residual_table.py formatting for agent totals."""
    if value is None:
        return "--"
    rounded = round(value)
    if rounded >= 1_000_000:
        return f"{rounded / 1_000_000:.2f}M"
    if rounded >= 1_000:
        return f"{rounded / 1_000:.0f}k"
    return str(rounded)


def fmt_usd(value):
    if value is None:
        return "--"
    return f"\\${value:.2f}"


def fmt_time_minutes(value):
    if value is None:
        return "--"
    return f"{value:.1f}"


def cell_cost(input_tokens, cached_input_tokens, output_tokens, reasoning_tokens, rates):
    if input_tokens is None and output_tokens is None and reasoning_tokens is None:
        return None
    input_tokens = input_tokens or 0
    cached_input_tokens = cached_input_tokens or 0
    output_tokens = output_tokens or 0
    reasoning_tokens = reasoning_tokens or 0
    non_cached = max(input_tokens - cached_input_tokens, 0)
    return (
        non_cached * rates["input"]
        + cached_input_tokens * rates["cached_input"]
        + output_tokens * rates["output"]
        + reasoning_tokens * rates.get("reasoning", rates["output"])
    ) / 1_000_000.0


def mutation_scores(mut_data, cov_data):
    return [
        comparison._mutation_score_or_zero_if_no_passing(mut_data, cov_data, lib)
        for lib in COMPARISON_LIBS
    ]


def coverage_means(cov_data):
    stmt = [cov_data[lib]["stmt_cov"] for lib in COMPARISON_LIBS]
    branch = [cov_data[lib]["branch_cov"] for lib in COMPARISON_LIBS]
    return mean(stmt), mean(branch)


def collect_testpilot_mean_time_minutes(model_dir):
    # Whole-suite mean over the 20-lib resource set (a resource descriptor, like
    # tokens/cost); fs-extra is excluded to match the other columns.
    run_dir = os.path.join(model_dir, "1")
    total_ms = 0.0
    for lib in RESOURCE_LIBS:
        path = os.path.join(run_dir, lib, "codexQueryTimes.json")
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            queries = json.load(fh)
        total_ms += sum(q[1] for q in queries)
    return total_ms / 60000.0 / len(RESOURCE_LIBS)


def testpilot_passing_loc(model_dir):
    """Whole-suite non-blank LoC of PASSED TestPilot test files (over RESOURCE_LIBS).

    A resource/volume descriptor paired with the passing-test count, computed over
    the 20-lib resource set (fs-extra excluded, matching the other columns)."""
    run_dir = os.path.join(model_dir, "1")
    total = 0
    for lib in RESOURCE_LIBS:
        report = comparison.read_json(os.path.join(run_dir, lib, "report.json"))
        if not report:
            continue
        tests_dir = os.path.join(run_dir, lib, "tests")
        seen = set()
        for test in report.get("tests", []):
            test_file = test.get("testFile")
            if not test_file or test_file in seen:
                continue
            seen.add(test_file)
            if test.get("status") == "PASSED":
                total += _nonblank_loc(os.path.join(tests_dir, test_file))
    return total


def agent_passing_loc(runs):
    """Mean over the HEAD runs (R3/R4) of whole-suite non-blank LoC of PASSED agent
    test files. Matches the passing-test count (mean of the same runs)."""
    per_run = []
    for run_id in agent_residual.RUN_IDS:
        run_dir = os.path.join(agent_residual.DATA_DIR, agent_residual.RUN_DIRS[run_id])
        total = 0
        for lib in RESOURCE_LIBS:
            report = agent_residual.read_json(
                os.path.join(run_dir, lib, "report", "report.json"))
            if not report:
                continue
            tests_dir = os.path.join(run_dir, lib, "report", "tests")
            seen = set()
            for test in report.get("tests", []):
                test_file = test.get("testFile")
                if not test_file or test_file in seen:
                    continue
                seen.add(test_file)
                if test.get("status") == "PASSED":
                    total += _nonblank_loc(os.path.join(tests_dir, test_file))
        per_run.append(total)
    return mean(per_run)


def collect_testpilot_rows():
    cov_500 = comparison.collect_coverage_stats(comparison.GPT54_500_DIR)
    cov_1000 = comparison.collect_coverage_stats(comparison.GPT54_1000_DIR)
    mut_500 = comparison.collect_mutation_scores(comparison.GPT54_500_DIR)
    mut_1000 = comparison.collect_mutation_scores(comparison.GPT54_1000_DIR)
    tokens_500 = token_table.collect_gpt5_tokens(token_table.GPT54_500_DIR)
    tokens_1000 = token_table.collect_gpt5_tokens(token_table.GPT54_1000_DIR)
    rates = agent_residual.load_pricing()

    scores_500 = mutation_scores(mut_500, cov_500)
    scores_1000 = mutation_scores(mut_1000, cov_1000)
    stmt_500, branch_500 = coverage_means(cov_500)
    stmt_1000, branch_1000 = coverage_means(cov_1000)
    input_500 = sum((tokens_500[lib]["input"] or 0) for lib in RESOURCE_LIBS if lib in tokens_500)
    output_500 = sum((tokens_500[lib]["output"] or 0) for lib in RESOURCE_LIBS if lib in tokens_500)
    input_1000 = sum((tokens_1000[lib]["input"] or 0) for lib in RESOURCE_LIBS if lib in tokens_1000)
    output_1000 = sum((tokens_1000[lib]["output"] or 0) for lib in RESOURCE_LIBS if lib in tokens_1000)

    return {
        "500": {
            "system": "GPT-5.4 TestPilot-500",
            "scores": scores_500,
            "stmt_coverage": stmt_500,
            "branch_coverage": branch_500,
            "passing_tests": sum(cov_500[lib]["passing"] or 0 for lib in RESOURCE_LIBS),
            "passing_loc": testpilot_passing_loc(comparison.GPT54_500_DIR),
            "input_tokens": input_500,
            "cached_input_tokens": 0,
            "output_tokens": output_500,
            "cost": cell_cost(input_500, 0, output_500, 0, rates),
            "time_minutes": collect_testpilot_mean_time_minutes(comparison.GPT54_500_DIR),
        },
        "1000": {
            "system": "GPT-5.4 TestPilot-1000",
            "scores": scores_1000,
            "stmt_coverage": stmt_1000,
            "branch_coverage": branch_1000,
            "passing_tests": sum(cov_1000[lib]["passing"] or 0 for lib in RESOURCE_LIBS),
            "passing_loc": testpilot_passing_loc(comparison.GPT54_1000_DIR),
            "input_tokens": input_1000,
            "cached_input_tokens": 0,
            "output_tokens": output_1000,
            "cost": cell_cost(input_1000, 0, output_1000, 0, rates),
            "time_minutes": collect_testpilot_mean_time_minutes(comparison.GPT54_1000_DIR),
        },
    }


def collect_agent_row():
    rates = agent_residual.load_pricing()
    runs = {run_id: agent_residual.collect_run(run_id) for run_id in agent_residual.RUN_IDS}
    rows = agent_residual.aggregate(runs, rates)
    scores = [rows[lib]["residual_score"] for lib in COMPARISON_LIBS]
    coverage = collect_agent_coverage(runs)

    return {
        "system": "Final agent",
        "scores": scores,
        "stmt_coverage": coverage["stmt_coverage"],
        "branch_coverage": coverage["branch_coverage"],
        "passing_tests": sum((rows[lib]["tests_passing"] or 0) for lib in RESOURCE_LIBS),
        "passing_loc": agent_passing_loc(runs),
        "input_tokens": sum((rows[lib]["input_tokens"] or 0) for lib in RESOURCE_LIBS),
        "cached_input_tokens": sum(
            (rows[lib]["cached_input_tokens"] or 0) for lib in RESOURCE_LIBS
        ),
        "output_tokens": sum((rows[lib]["output_tokens"] or 0) for lib in RESOURCE_LIBS),
        "cost": sum((rows[lib]["cost"] or 0) for lib in RESOURCE_LIBS),
        "time_minutes": (
            sum((rows[lib]["duration_ms"] or 0) for lib in RESOURCE_LIBS)
            / 60000.0
            / len(RESOURCE_LIBS)
        ),
    }


def collect_agent_coverage(runs):
    per_lib = []
    for lib in COMPARISON_LIBS:
        lib_values = []
        for run_id in agent_residual.RUN_IDS:
            cell = runs[run_id].get(lib)
            if not cell:
                continue
            if cell.get("no_test_score"):
                lib_values.append((0.0, 0.0))
                continue
            if not cell.get("score_measurement_complete"):
                continue
            report_path = os.path.join(
                agent_residual.DATA_DIR, agent_residual.RUN_DIRS[run_id],
                lib, "report", "report.json"
            )
            report = agent_residual.read_json(report_path)
            if report is None:
                continue
            total = report.get("coverage", {}).get("total", {})
            stmt = total.get("statements", {}).get("pct")
            branch = total.get("branches", {}).get("pct")
            if isinstance(stmt, (int, float)) and isinstance(branch, (int, float)):
                lib_values.append((stmt, branch))
        if not lib_values:
            continue
        per_lib.append((
            mean([v[0] for v in lib_values]),
            mean([v[1] for v in lib_values]),
        ))
    return {
        "stmt_coverage": mean([v[0] for v in per_lib]),
        "branch_coverage": mean([v[1] for v in per_lib]),
        "coverage_n": len(per_lib),
    }


def build_metrics():
    testpilot = collect_testpilot_rows()
    agent = collect_agent_row()

    agent_scores = agent["scores"]
    for key, row in testpilot.items():
        row["agent_delta_mean"] = mean(
            [agent_score - testpilot_score for agent_score, testpilot_score in zip(agent_scores, row["scores"])]
        )
    agent["agent_delta_mean"] = None

    rows = [testpilot["500"], testpilot["1000"], agent]
    for row in rows:
        row["mutation_n"] = len(row["scores"])
        row["mutation_mean"] = mean(row["scores"])
        row["mutation_median"] = statistics.median(row["scores"])

    return rows, agent


def validate(rows, agent):
    # Golden regression guard: expectations below match the shipped data
    # snapshot and sys.exit on mismatch. Update them deliberately when the
    # data is regenerated.
    by_system = {row["system"]: row for row in rows}
    # Golden values for the HEAD agent runs (R3/R4). All columns (score, coverage, and
    # the resource descriptors: passing tests, LoC, tokens, cost, time) are over the
    # 20-library set excluding fs-extra (COMPARISON_LIBS / RESOURCE_LIBS).
    # 2026-06-27a: TestPilot-side score/coverage refreshed after splicing the fixed-harness
    #   crawler-url-parser runs (was a 0% no-passing-tests artifact; now real data).
    # 2026-06-27b: fs-extra excluded from the score/coverage aggregates (overfit validation
    #   artifact), dropping the comparison set 21 -> 20. This lifts TestPilot/agent coverage
    #   and TestPilot mutation slightly and trims the agent's mutation-score advantage
    #   (+12.7/+9.3 -> +12.1/+8.2).
    # 2026-06-28: fs-extra also dropped from the resource columns so every column is 20-lib
    #   and matches the 20-lib token table (fs-extra was a large agent cost: ~18M input
    #   tokens, ~$7 of the agent total).
    expected = {
        "comparison_n": 20,
        "tp500_mean_1dp": 54.4,
        "tp500_median_1dp": 62.4,
        "tp1000_mean_1dp": 58.3,
        "tp1000_median_1dp": 59.9,
        "agent_mean_1dp": 66.5,
        "agent_median_1dp": 75.5,
        "delta_vs_500_1dp": 12.1,
        "delta_vs_1000_1dp": 8.2,
        "tp500_stmt_1dp": 79.9,
        "tp500_branch_1dp": 68.7,
        "tp1000_stmt_1dp": 84.9,
        "tp1000_branch_1dp": 74.3,
        "agent_stmt_1dp": 78.6,
        "agent_branch_1dp": 71.3,
        "tp500_passing": 3057,
        "tp1000_passing": 3246,
        "agent_passing": 826,
        "tp500_passloc": 84745,
        "tp1000_passloc": 94182,
        "agent_passloc": 40251,
        "tp500_input": 2374538,
        "tp500_output": 6912557,
        "tp1000_input": 2607490,
        "tp1000_output": 7221243,
        "agent_input_rounded": 72076294,
        "agent_cached_rounded": 64826816,
        "agent_output_rounded": 1516998,
        "tp500_cost_2dp": 72.09,
        "tp1000_cost_2dp": 75.47,
        "agent_cost_2dp": 32.34,
        "tp500_time_1dp": 24.7,
        "tp1000_time_1dp": 29.2,
        "agent_time_1dp": 35.7,
    }
    actual = {
        "comparison_n": len(COMPARISON_LIBS),
        "tp500_mean_1dp": round(by_system["GPT-5.4 TestPilot-500"]["mutation_mean"], 1),
        "tp500_median_1dp": round(by_system["GPT-5.4 TestPilot-500"]["mutation_median"], 1),
        "tp1000_mean_1dp": round(by_system["GPT-5.4 TestPilot-1000"]["mutation_mean"], 1),
        "tp1000_median_1dp": round(by_system["GPT-5.4 TestPilot-1000"]["mutation_median"], 1),
        "agent_mean_1dp": round(by_system["Final agent"]["mutation_mean"], 1),
        "agent_median_1dp": round(by_system["Final agent"]["mutation_median"], 1),
        "delta_vs_500_1dp": round(by_system["GPT-5.4 TestPilot-500"]["agent_delta_mean"], 1),
        "delta_vs_1000_1dp": round(by_system["GPT-5.4 TestPilot-1000"]["agent_delta_mean"], 1),
        "tp500_stmt_1dp": round(by_system["GPT-5.4 TestPilot-500"]["stmt_coverage"], 1),
        "tp500_branch_1dp": round(by_system["GPT-5.4 TestPilot-500"]["branch_coverage"], 1),
        "tp1000_stmt_1dp": round(by_system["GPT-5.4 TestPilot-1000"]["stmt_coverage"], 1),
        "tp1000_branch_1dp": round(by_system["GPT-5.4 TestPilot-1000"]["branch_coverage"], 1),
        "agent_stmt_1dp": round(by_system["Final agent"]["stmt_coverage"], 1),
        "agent_branch_1dp": round(by_system["Final agent"]["branch_coverage"], 1),
        "tp500_passing": round(by_system["GPT-5.4 TestPilot-500"]["passing_tests"]),
        "tp1000_passing": round(by_system["GPT-5.4 TestPilot-1000"]["passing_tests"]),
        "agent_passing": round(by_system["Final agent"]["passing_tests"]),
        "tp500_passloc": round(by_system["GPT-5.4 TestPilot-500"]["passing_loc"]),
        "tp1000_passloc": round(by_system["GPT-5.4 TestPilot-1000"]["passing_loc"]),
        "agent_passloc": round(by_system["Final agent"]["passing_loc"]),
        "tp500_input": by_system["GPT-5.4 TestPilot-500"]["input_tokens"],
        "tp500_output": by_system["GPT-5.4 TestPilot-500"]["output_tokens"],
        "tp1000_input": by_system["GPT-5.4 TestPilot-1000"]["input_tokens"],
        "tp1000_output": by_system["GPT-5.4 TestPilot-1000"]["output_tokens"],
        "agent_input_rounded": round(by_system["Final agent"]["input_tokens"]),
        "agent_cached_rounded": round(by_system["Final agent"]["cached_input_tokens"]),
        "agent_output_rounded": round(by_system["Final agent"]["output_tokens"]),
        "tp500_cost_2dp": round(by_system["GPT-5.4 TestPilot-500"]["cost"], 2),
        "tp1000_cost_2dp": round(by_system["GPT-5.4 TestPilot-1000"]["cost"], 2),
        "agent_cost_2dp": round(by_system["Final agent"]["cost"], 2),
        "tp500_time_1dp": round(by_system["GPT-5.4 TestPilot-500"]["time_minutes"], 1),
        "tp1000_time_1dp": round(by_system["GPT-5.4 TestPilot-1000"]["time_minutes"], 1),
        "agent_time_1dp": round(by_system["Final agent"]["time_minutes"], 1),
    }

    discrepancies = []
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if actual_value != expected_value:
            discrepancies.append((key, expected_value, actual_value))

    if discrepancies:
        print("Validation failed; not writing thesis-facing table.")
        for key, expected_value, actual_value in discrepancies:
            print(f"  {key}: expected {expected_value}, got {actual_value}")
        sys.exit(1)


def latex_row(row):
    if row["system"].startswith("GPT-5.4 TestPilot"):
        fmt_token = fmt_testpilot_token
    else:
        fmt_token = fmt_agent_token

    return (
        f"{DISPLAY_NAME[row['system']]} & "
        f"{pct(row['mutation_mean'])}/{pct(row['mutation_median'])} & "
        f"{pct(row['stmt_coverage'])}/{pct(row['branch_coverage'])} & "
        f"{fmt_count(row['passing_tests'])} & "
        f"{fmt_loc(row['passing_loc'])} & "
        f"{fmt_token(row['input_tokens'])} & "
        f"{fmt_token(row['cached_input_tokens'])} & "
        f"{fmt_token(row['output_tokens'])} & "
        f"{fmt_usd(row['cost'])} & "
        f"{fmt_time_minutes(row['time_minutes'])} \\\\"
    )


def write_tex(rows, agent):
    lines = [
        WARN,
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Comparison between GPT-5.4 TestPilot and the final GPT-5.4 agent. "
        r"\emph{Mut.}\ is mutation score (mean/median over libraries); \emph{Cov.}\ is coverage "
        r"(statement/branch); \emph{Tests} and \emph{LoC} are the passing test count and the "
        r"lines of passing test code; \emph{In}/\emph{Cached}/\emph{Out} are token "
        r"totals. \emph{Min/lib} is mean minutes per library.}",
        r"\label{tab:gpt5-agent-testpilot-comparison}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"\textbf{System} & \textbf{Mut.} & \textbf{Cov.} & \textbf{Tests} & "
        r"\textbf{LoC} & \textbf{In} & \textbf{Cached} & \textbf{Out} & "
        r"\textbf{Cost} & \textbf{Min/lib} \\",
        r"\midrule",
    ]
    lines.extend(latex_row(row) for row in rows)
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\par\vspace{0.35em}",
        r"{\footnotesize\raggedright \textit{Note.} All columns are computed over the "
        r"20-library set, excluding \texttt{fs-extra}: its TestPilot tests overfit to "
        r"TestPilot's working directory (a validation artifact discussed in "
        r"Section~\ref{sec:results:testpilot-validation-issues}), so its scores do not "
        r"reflect genuine test quality.\par}",
        r"\end{table}",
        "",
    ])
    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(OUTPUT_TEX, "w") as fh:
        fh.write("\n".join(lines))


def write_csv(rows, agent):
    fields = [
        "system",
        "mutation_mean",
        "mutation_median",
        "agent_delta_mean_pp",
        "stmt_coverage",
        "branch_coverage",
        "passing_tests",
        "passing_loc",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "cost_usd",
        "time_minutes",
        "comparison_set",
    ]
    comparison_set = ",".join(COMPARISON_LIBS)
    with open(OUTPUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "system": row["system"],
                "mutation_mean": row["mutation_mean"],
                "mutation_median": row["mutation_median"],
                "agent_delta_mean_pp": row["agent_delta_mean"],
                "stmt_coverage": row["stmt_coverage"],
                "branch_coverage": row["branch_coverage"],
                "passing_tests": row["passing_tests"],
                "passing_loc": round(row["passing_loc"]),
                "input_tokens": row["input_tokens"],
                "cached_input_tokens": row["cached_input_tokens"],
                "output_tokens": row["output_tokens"],
                "cost_usd": row["cost"],
                "time_minutes": row["time_minutes"],
                "comparison_set": comparison_set,
            })


# ---------------------------------------------------------------------------
# Per-library detail table (coverage + cost), for the appendix
# ---------------------------------------------------------------------------

def _mean_or_none(values):
    present = [v for v in values if v is not None]
    return (sum(present) / len(present)) if present else None


def agent_per_lib_coverage(runs):
    """Per-library (statement, branch) agent coverage, mean over the HEAD runs.

    Mirrors collect_agent_coverage() exactly but keeps the per-library values
    instead of averaging across libraries, so the column means below equal the
    aggregate agent coverage in Table~\\ref{tab:gpt5-agent-testpilot-comparison}."""
    out = {}
    for lib in COMPARISON_LIBS:
        lib_values = []
        for run_id in agent_residual.RUN_IDS:
            cell = runs[run_id].get(lib)
            if not cell:
                continue
            if cell.get("no_test_score"):
                lib_values.append((0.0, 0.0))
                continue
            if not cell.get("score_measurement_complete"):
                continue
            report = agent_residual.read_json(os.path.join(
                agent_residual.DATA_DIR, agent_residual.RUN_DIRS[run_id],
                lib, "report", "report.json"))
            if report is None:
                continue
            total = report.get("coverage", {}).get("total", {})
            stmt = total.get("statements", {}).get("pct")
            branch = total.get("branches", {}).get("pct")
            if isinstance(stmt, (int, float)) and isinstance(branch, (int, float)):
                lib_values.append((stmt, branch))
        out[lib] = ((mean([v[0] for v in lib_values]),
                     mean([v[1] for v in lib_values])) if lib_values else None)
    return out


# Column order shared by the per-library row builder, writer, and validation.
PERLIB_KEYS = ["tp500_mut", "tp500_stmt", "tp500_branch", "tp500_cost",
               "tp1000_mut", "tp1000_stmt", "tp1000_branch", "tp1000_cost",
               "agent_mut", "agent_stmt", "agent_branch", "agent_cost"]


def collect_per_library_rows():
    rates = agent_residual.load_pricing()
    cov_500 = comparison.collect_coverage_stats(comparison.GPT54_500_DIR)
    cov_1000 = comparison.collect_coverage_stats(comparison.GPT54_1000_DIR)
    mut_500 = comparison.collect_mutation_scores(comparison.GPT54_500_DIR)
    mut_1000 = comparison.collect_mutation_scores(comparison.GPT54_1000_DIR)
    tokens_500 = token_table.collect_gpt5_tokens(token_table.GPT54_500_DIR)
    tokens_1000 = token_table.collect_gpt5_tokens(token_table.GPT54_1000_DIR)

    runs = {r: agent_residual.collect_run(r) for r in agent_residual.RUN_IDS}
    agg = agent_residual.aggregate(runs, rates)
    agent_cov = agent_per_lib_coverage(runs)

    def tp_cost(tokens, lib):
        # cell_cost is linear in tokens, so summing per-library costs reproduces
        # the whole-suite TestPilot cost in the aggregate table exactly.
        if lib not in tokens:
            return None
        return cell_cost(tokens[lib].get("input"), 0, tokens[lib].get("output"), 0, rates)

    rows = []
    for lib in COMPARISON_LIBS:
        acov = agent_cov.get(lib)
        rows.append({
            "lib": lib,
            "tp500_mut": comparison._mutation_score_or_zero_if_no_passing(mut_500, cov_500, lib),
            "tp500_stmt": cov_500.get(lib, {}).get("stmt_cov"),
            "tp500_branch": cov_500.get(lib, {}).get("branch_cov"),
            "tp500_cost": tp_cost(tokens_500, lib),
            "tp1000_mut": comparison._mutation_score_or_zero_if_no_passing(mut_1000, cov_1000, lib),
            "tp1000_stmt": cov_1000.get(lib, {}).get("stmt_cov"),
            "tp1000_branch": cov_1000.get(lib, {}).get("branch_cov"),
            "tp1000_cost": tp_cost(tokens_1000, lib),
            "agent_mut": agg[lib]["residual_score"] if agg.get(lib, {}).get("residual_score") is not None else None,
            "agent_stmt": acov[0] if acov else None,
            "agent_branch": acov[1] if acov else None,
            "agent_cost": agg[lib]["cost"] if agg.get(lib, {}).get("cost") is not None else None,
        })
    return rows


def validate_per_library(rows):
    """Tie the per-library table back to the aggregate goldens: column coverage
    means and cost totals must match Table~\\ref{tab:gpt5-agent-testpilot-comparison}."""
    def cov_mean(key):
        return round(_mean_or_none([r[key] for r in rows]), 1)

    def cost_total(key):
        return round(sum(r[key] or 0 for r in rows), 2)

    expected = {
        "n": 20,
        "tp500_mut": 54.4, "tp1000_mut": 58.3, "agent_mut": 66.5,
        "tp500_stmt": 79.9, "tp500_branch": 68.7,
        "tp1000_stmt": 84.9, "tp1000_branch": 74.3,
        "agent_stmt": 78.6, "agent_branch": 71.3,
        "tp500_cost_total": 72.09, "tp1000_cost_total": 75.47, "agent_cost_total": 32.34,
    }
    actual = {
        "n": len(rows),
        "tp500_mut": cov_mean("tp500_mut"), "tp1000_mut": cov_mean("tp1000_mut"),
        "agent_mut": cov_mean("agent_mut"),
        "tp500_stmt": cov_mean("tp500_stmt"), "tp500_branch": cov_mean("tp500_branch"),
        "tp1000_stmt": cov_mean("tp1000_stmt"), "tp1000_branch": cov_mean("tp1000_branch"),
        "agent_stmt": cov_mean("agent_stmt"), "agent_branch": cov_mean("agent_branch"),
        "tp500_cost_total": cost_total("tp500_cost"),
        "tp1000_cost_total": cost_total("tp1000_cost"),
        "agent_cost_total": cost_total("agent_cost"),
    }
    discrepancies = [(k, expected[k], actual[k]) for k in expected if actual[k] != expected[k]]
    if discrepancies:
        print("Per-library validation failed; not writing detail table.")
        for k, exp, act in discrepancies:
            print(f"  {k}: expected {exp}, got {act}")
        sys.exit(1)


def write_per_library_tex(rows):
    def cov(v):
        return f"{v:.1f}" if v is not None else "--"

    def usd(v):
        return f"\\${v:.2f}" if v is not None else "--"

    def cell(key, v):
        return usd(v) if key.endswith("cost") else cov(v)

    lines = [
        WARN,
        r"\begin{table*}[!htb]",
        r"\centering",
        r"\caption{Per-library mutation score, statement and branch coverage, and generation "
        r"cost for the two GPT-5.4 \textsc{TestPilot} budgets and the final GPT-5.4 agent. "
        r"\emph{Mut}, \emph{Stmt}, and \emph{Br} are mutation score and statement and branch "
        r"coverage in percent. \emph{Cost} is the USD generation cost for that library. "
        r"Failing to generate any passing tests counts 0.0\% mutation score. Agent values "
        r"are means over the two temperature 1 runs, and the agent mutation score is the "
        r"final (post-residual) score.}",
        r"\label{tab:gpt5-agent-testpilot-per-library}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l rrrr rrrr rrrr}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Project}} & "
        r"\multicolumn{4}{c}{\textbf{TestPilot-500}} & "
        r"\multicolumn{4}{c}{\textbf{TestPilot-1000}} & "
        r"\multicolumn{4}{c}{\textbf{Agent}} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}",
        r" & \textbf{Mut} & \textbf{Stmt} & \textbf{Br} & \textbf{Cost}"
        r" & \textbf{Mut} & \textbf{Stmt} & \textbf{Br} & \textbf{Cost}"
        r" & \textbf{Mut} & \textbf{Stmt} & \textbf{Br} & \textbf{Cost} \\",
        r"\midrule",
    ]
    for r in rows:
        cells = " & ".join(cell(k, r[k]) for k in PERLIB_KEYS)
        lines.append(f"{r['lib']} & {cells} \\\\")
        if r["lib"] in comparison.SEPARATOR_LIBS:
            lines.append(r"\midrule")
    lines.append(r"\midrule")
    means = {k: _mean_or_none([r[k] for r in rows]) for k in PERLIB_KEYS}
    lines.append(r"\textbf{Mean} & " + " & ".join(cell(k, means[k]) for k in PERLIB_KEYS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}", ""]
    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(OUTPUT_PERLIB_TEX, "w") as fh:
        fh.write("\n".join(lines))


def write_per_library_csv(rows):
    fields = ["library"] + PERLIB_KEYS
    with open(OUTPUT_PERLIB_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({"library": r["lib"], **{k: r[k] for k in PERLIB_KEYS}})


def main():
    rows, agent = build_metrics()
    validate(rows, agent)
    write_tex(rows, agent)
    write_csv(rows, agent)

    per_library = collect_per_library_rows()
    validate_per_library(per_library)
    write_per_library_tex(per_library)
    write_per_library_csv(per_library)
    print(f"wrote {os.path.relpath(OUTPUT_PERLIB_TEX, SCRIPT_DIR)}")
    print(f"wrote {os.path.relpath(OUTPUT_PERLIB_CSV, SCRIPT_DIR)}")

    print(f"wrote {os.path.relpath(OUTPUT_TEX, SCRIPT_DIR)}")
    print(f"wrote {os.path.relpath(OUTPUT_CSV, SCRIPT_DIR)}")
    print(f"comparison set size: {len(COMPARISON_LIBS)}")
    for row in rows:
        print(
            f"{row['system']}: mean {row['mutation_mean']:.2f}%, "
            f"median {row['mutation_median']:.2f}%, "
            f"stmt {row['stmt_coverage']:.2f}%, branch {row['branch_coverage']:.2f}%, "
            f"passing tests {row['passing_tests']:.0f}, "
            f"mean time {row['time_minutes']:.1f} min"
        )
    print(
        "Final agent totals: "
        f"{agent['input_tokens']:.0f} input, "
        f"{agent['cached_input_tokens']:.0f} cached, "
        f"{agent['output_tokens']:.0f} output, "
        f"${agent['cost']:.2f}, "
        f"{agent['time_minutes']:.1f} mean min/library"
    )


if __name__ == "__main__":
    main()
