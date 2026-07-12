#!/usr/bin/env python3
"""
Generate LaTeX comparison tables for GPT-Turbo vs GPT-5.4-output-500, GPT-5.4-output-1000,
and GPT-5.4-agent.

Tables produced (written flat to ./tables/; the thesis/Overleaf project sorts
them into testpilot/ and pilot/ subdirs — see the grouping below):
  tab-gpt5-stmt-cov-comparison.tex          – statement coverage per library     [Overleaf: testpilot/]
  tab-gpt5-branch-cov-comparison.tex        – branch coverage per library
  tab-gpt5-mutation-comparison.tex          – mutation score + tests discarded during dry run [Overleaf: pilot/]
  tab-gpt5-testpilot-mutation-comparison.tex – mutation score without agent columns [Overleaf: testpilot/]
  tab-gpt5-500-general-mutation-score.tex   – standalone mutation score table for GPT-5.4-500
  tab-gpt5-1000-general-mutation-score.tex  – standalone mutation score table for GPT-5.4-1000
  tab-gpt5-agent-general-mutation-score.tex – standalone mutation score table for GPT-5.4-agent [Overleaf: pilot/]

IMPORTANT — two output families, one script:
  * testpilot/ tables (stmt-cov-comparison, testpilot-mutation-comparison) have
    NO agent column: they compare GPT-Turbo vs the GPT-5.4 TestPilot output-token
    variants only.
  * pilot/ tables (mutation-comparison, agent-general-mutation-score) DO include
    an agent column, sourced from data/gpt-5.4-agent/ — the single OLD PILOT run
    that PREDATES the mutation-feedback/residual flags (a different, older agent
    commit than the 63be77f R3/R4 set used by the final result tables). These are
    intentionally the pilot numbers that motivated later agent changes; they are
    NOT the same-SHA result tables in generate_gpt5_agent_residual_table.py.
  This script is shared by both families, so it cannot be deleted as "pilot-only".

Metrics:
  - Coverage values come from report.json > coverage.total.{statements,branches}.pct
  - Mutation scores come from report.json > mutationMetrics.all.mutationScore (gpt-5.4 models)
    or are computed from mutation-report.json (killed / (killed + survived) for gptturbo)
    or from report/mutation/mutation.json (agent)
  - Missing mutation scores caused by zero passing tests are
    treated as 0.0% in the mutation-score tables and aggregate medians. Missing
    scores for other reasons remain unavailable ("--").
  - Discarded tests = tests with status==PASSED but dryRunStatus==failed in report.json

For GPT-Turbo (10 runs) the median across runs is used; GPT-5.4 models have a single run.
"""

import json
import os
import statistics

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
TABLES_DIR = os.path.join(SCRIPT_DIR, "tables")

GPTTURBO_DIR    = os.path.join(DATA_DIR, "gptturbo")
GPT54_500_DIR   = os.path.join(DATA_DIR, "gpt-5.4-output-500")
GPT54_1000_DIR  = os.path.join(DATA_DIR, "gpt-5.4-output-1000")
GPT54_AGENT_DIR = os.path.join(DATA_DIR, "gpt-5.4-agent")

# Mirrors domainMap order in utils.js
DOMAIN_MAP = [
    "glob", "fs-extra", "graceful-fs", "jsonfile",
    "q", "node-dir", "zip-a-folder",
    "quill-delta", "complex.js", "pull-stream",
    "countries-and-timezones", "simple-statistics", "plural", "dirty",
    "geo-point", "uneval", "image-downloader", "crawler-url-parser",
    "gitlab-js", "core", "omnitool",
]

SEPARATOR_LIBS = {"zip-a-folder", "uneval"}

# Libraries excluded from the mutation-score tables because they have no usable
# GPT-Turbo mutation baseline — for two DISTINCT reasons, kept as separate sets so
# the reason each lib is out is explicit in code:
#
#   NODE_INCOMPATIBLE_LIBS — incompatible with the current Node.js version, so GPT-5.4
#     never generated for them (they are not even in DOMAIN_MAP). No GPT-5.4 data at all.
#
#   OVERFIT_EXCLUDED_LIBS — fs-extra runs fine and produces coverage on every modern
#     system, but TestPilot's RetryWithError refinement makes its generated tests depend
#     on TestPilot's own working directory (e.g. asserting the file count of TestPilot's
#     test/ folder). They pass under TestPilot (and count toward coverage) but fail in the
#     Stryker sandbox, so GPT-Turbo has no fs-extra mutation score. Because that result is
#     a validation artifact, fs-extra is also excluded from the cross-system SCORE/coverage
#     aggregates of the agent-vs-TestPilot comparison (see generate_gpt5_agent_testpilot_
#     comparison_table.py). It is now ALSO dropped entirely from the coverage comparison
#     (previously kept and daggered), as fs-extra is excluded thesis-wide for now.
NODE_INCOMPATIBLE_LIBS = {"rsvp", "bluebird", "memfs", "js-sdsl"}
OVERFIT_EXCLUDED_LIBS = {"fs-extra"}
NO_MUTATION_BASELINE_LIBS = NODE_INCOMPATIBLE_LIBS | OVERFIT_EXCLUDED_LIBS

# Libraries dropped entirely from the coverage comparison tables (and from the mutation
# tables via NO_MUTATION_BASELINE_LIBS). fs-extra is the environment-overfitting case.
COVERAGE_EXCLUDED_LIBS = OVERFIT_EXCLUDED_LIBS

# Libraries flagged with a dagger as problematic replication/validation cases.
#   (none) — fs-extra was the only daggered case and is now dropped from the coverage
#   comparison entirely (see COVERAGE_EXCLUDED_LIBS), so no rows carry a dagger.
# crawler-url-parser was previously flagged (generated tests failed to run, "Invalid
# syntax", 0 passing tests). It has since been regenerated with the corrected closeBrackets
# validator (data spliced 2026-06-27, fixed-harness runs), so it now has genuine coverage
# and mutation results and is no longer flagged. graceful-fs (measurement noise,
# documented/resolved) and image-downloader (an ordinary poor model result, not a
# validation artifact) keep their rows but are NOT flagged.
MUTATION_PROBLEM_LIBS = set()
COVERAGE_PROBLEM_LIBS = set()

VALIDATION_NOTE = (
    r"\parbox{\textwidth}{\footnotesize\textit{Note.} Projects marked with "
    r"$^{\dagger}$ are discussed as problematic replication or validation cases "
    r"in Section~\ref{sec:results:testpilot-validation-issues}.}"
)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _med(values):
    return statistics.median(values) if values else None


def _runs(model_dir):
    return sorted(
        e for e in os.listdir(model_dir)
        if not e.startswith(".") and os.path.isdir(os.path.join(model_dir, e))
    )


def collect_coverage_stats(model_dir):
    """
    Returns {lib: {stmt_cov, branch_cov, discarded, passing}}
    where each value is the median across all runs in model_dir.
    """
    acc = {}  # {lib: {key: [values]}}

    for run in _runs(model_dir):
        run_dir = os.path.join(model_dir, run)
        for lib in os.listdir(run_dir):
            if lib.startswith(".") or lib not in DOMAIN_MAP:
                continue
            report = read_json(os.path.join(run_dir, lib, "report.json"))
            if report is None:
                continue

            cov = report.get("coverage", {}).get("total", {})
            stmt_pct = cov.get("statements", {}).get("pct")
            branch_pct = cov.get("branches", {}).get("pct")
            line_pct = cov.get("lines", {}).get("pct")

            tests = report.get("tests", [])
            passing = sum(1 for t in tests if t.get("status") == "PASSED")
            discarded = sum(
                1 for t in tests
                if t.get("status") == "PASSED" and t.get("dryRunStatus") == "failed"
            )

            entry = acc.setdefault(lib, {"stmt": [], "branch": [], "line": [], "discarded": [], "passing": []})
            if stmt_pct is not None:
                entry["stmt"].append(stmt_pct)
            if branch_pct is not None:
                entry["branch"].append(branch_pct)
            if line_pct is not None:
                entry["line"].append(line_pct)
            entry["discarded"].append(discarded)
            entry["passing"].append(passing)

    return {
        lib: {
            "stmt_cov":  _med(v["stmt"]),
            "branch_cov": _med(v["branch"]),
            "line_cov":  _med(v["line"]),
            "discarded": _med(v["discarded"]),
            "passing":   _med(v["passing"]),
        }
        for lib, v in acc.items()
    }


def _mutation_score_from_report(report):
    """Read pre-computed mutation score from report.json (gpt-5.4 models)."""
    mm = report.get("mutationMetrics", {})
    return mm.get("all", {}).get("mutationScore") if mm else None


def _mutation_score_from_stryker_json(data):
    """Compute mutation score from a parsed Stryker JSON report (killed / (killed+survived))."""
    killed = survived = 0
    for file_data in data.get("files", {}).values():
        for mutant in file_data.get("mutants", []):
            status = mutant.get("status", "")
            if status == "Killed":
                killed += 1
            elif status == "Survived":
                survived += 1
    total = killed + survived
    return (killed / total * 100) if total > 0 else None


def _mutation_score_from_mutation_report(lib_dir):
    """Compute mutation score from mutation-report.json (killed / (killed+survived))."""
    mutation_report = read_json(os.path.join(lib_dir, "mutation-report.json"))
    if mutation_report is None:
        return None
    return _mutation_score_from_stryker_json(mutation_report)


def _mutation_score_from_mutation_report_file(path):
    """Compute mutation score from an explicit Stryker JSON file path."""
    data = read_json(path)
    if data is None:
        return None
    return _mutation_score_from_stryker_json(data)


def _mutation_score_or_zero_if_no_passing(mut_data, cov_data, lib):
    """Return mutation score, or 0.0 when the run produced no passing tests."""
    score = mut_data.get(lib)
    if score is not None:
        return score

    cov = cov_data.get(lib, {}) if cov_data is not None else {}
    if cov.get("passing") == 0:
        return 0.0
    return None


def collect_mutation_scores(model_dir):
    """
    Returns {lib: mutation_score_pct} (median across runs).
    Tries report.json mutationMetrics first, falls back to mutation-report.json.
    """
    acc = {}

    for run in _runs(model_dir):
        run_dir = os.path.join(model_dir, run)
        for lib in os.listdir(run_dir):
            if lib.startswith(".") or lib not in DOMAIN_MAP:
                continue
            lib_dir = os.path.join(run_dir, lib)
            report = read_json(os.path.join(lib_dir, "report.json"))

            score = None
            if report is not None:
                score = _mutation_score_from_report(report)
            if score is None:
                score = _mutation_score_from_mutation_report(lib_dir)

            if score is not None:
                acc.setdefault(lib, []).append(score)

    return {lib: _med(vals) for lib, vals in acc.items()}


def collect_agent_coverage_stats(model_dir):
    """
    Returns {lib: {stmt_cov, branch_cov, passing, discarded}} for the agent run.
    Reads from {lib}/report/report.json (agent-specific path).
    discarded is always None — agent reports have no dryRunStatus.
    """
    result = {}
    run_dir = os.path.join(model_dir, "1")
    if not os.path.isdir(run_dir):
        return result

    for lib in os.listdir(run_dir):
        if lib.startswith(".") or lib not in DOMAIN_MAP:
            continue
        report = read_json(os.path.join(run_dir, lib, "report", "report.json"))
        if report is None:
            continue

        cov = report.get("coverage", {}).get("total", {})
        stmt_pct   = cov.get("statements", {}).get("pct")
        branch_pct = cov.get("branches",   {}).get("pct")

        # pct can be the string "Unknown" when no instrumentation data
        if isinstance(stmt_pct,   str): stmt_pct   = None
        if isinstance(branch_pct, str): branch_pct = None

        tests   = report.get("tests", [])
        passing = sum(1 for t in tests if t.get("status") == "PASSED")

        # Match TestPilot methodology: only report coverage from passing tests.
        # The benchmark runs all tests in one NYC batch, so coverage includes
        # contributions from failing tests.  Suppress it when nothing passed.
        if passing == 0:
            stmt_pct   = None
            branch_pct = None

        result[lib] = {
            "stmt_cov":   stmt_pct,
            "branch_cov": branch_pct,
            "passing":    passing,
            "discarded":  None,
        }

    return result


def collect_agent_mutation_scores(model_dir, agent_cov_data=None):
    """
    Returns {lib: mutation_score_pct} for the agent run (single run, run ID "1").
    Reads from {lib}/report/mutation/mutation.json (Stryker JSON report).

    When *agent_cov_data* is provided (from collect_agent_coverage_stats), libraries
    with 0 passing tests are excluded.  This guards against the case where Stryker
    succeeded (e.g. because it uses --no-config) while the benchmark mocha did not.
    """
    result = {}
    run_dir = os.path.join(model_dir, "1")
    if not os.path.isdir(run_dir):
        return result

    for lib in os.listdir(run_dir):
        if lib.startswith(".") or lib not in DOMAIN_MAP:
            continue

        # Skip libraries where the benchmark recorded 0 passing tests.
        if agent_cov_data is not None:
            passing = agent_cov_data.get(lib, {}).get("passing")
            if passing is not None and passing == 0:
                continue

        mutation_file = os.path.join(run_dir, lib, "report", "mutation", "mutation.json")
        score = _mutation_score_from_mutation_report_file(mutation_file)
        if score is not None:
            result[lib] = score

    return result


# ---------------------------------------------------------------------------
# LaTeX formatting helpers
# ---------------------------------------------------------------------------

def lp(val, decimals=1):
    """Format a float as a LaTeX percentage, or -- if None."""
    if val is None:
        return "--"
    return f"{val:.{decimals}f}\\%"


def diff_value(val, ref):
    if val is None or ref is None:
        return None
    return val - ref


def delta_cell(diff, decimals=1):
    """Format an absolute percentage-point difference. Positive differences are bold."""
    if diff is None:
        return "--"
    sign = "+" if diff >= 0 else ""
    formatted = f"{sign}{diff:.{decimals}f}"
    return f"\\textbf{{{formatted}}}" if diff > 0 else formatted


def discard_cell(discarded, passing):
    """Format discarded tests as 'N (X%)'."""
    if discarded is None or passing is None:
        return "--"
    d = round(discarded)
    if passing == 0:
        return f"{d} (--)"
    pct = discarded / passing * 100
    return f"{d} ({pct:.1f}\\%)"


def write_table(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(content)
    print(f"Written: {path}")


def project_cell(lib, marked_libs=None):
    if marked_libs and lib in marked_libs:
        return f"{lib}$^{{\\dagger}}$"
    return lib


def maybe_add_validation_note(lines, marked_libs):
    if marked_libs:
        lines.append(r"\vspace{0.35em}")
        lines.append(VALIDATION_NOTE)


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------

def generate_cov_table(cov_turbo, cov_500, cov_1000, metric, label, caption,
                       output_path, marked_libs=None):
    """
    Generic coverage comparison table (used for both stmt cov and branch cov).
    metric: "stmt_cov" | "branch_cov"
    """
    lines = []
    lines.append(r"%WARNING: auto-generated. Edit generate_gpt5_comparison_tables.py instead.")
    lines.append(r"\begin{table*}[!htb]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"\multirow{2}{*}{\textbf{Project}} & "
        r"\multirow{2}{*}{\textbf{GPT-Turbo}} & "
        r"\multicolumn{2}{c}{\textbf{GPT-5.4-500}} & "
        r"\multicolumn{2}{c}{\textbf{GPT-5.4-1000}} \\"
    )
    lines.append(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}")
    lines.append(
        r" & & \textbf{Score} & \textbf{$\Delta$ (pp)} "
        r"& \textbf{Score} & \textbf{$\Delta$ (pp)} \\"
    )
    lines.append(r"\midrule")

    turbo_vals, s500_vals, s1000_vals = [], [], []
    s500_deltas, s1000_deltas = [], []

    for lib in DOMAIN_MAP:
        if lib in COVERAGE_EXCLUDED_LIBS:
            continue
        t = cov_turbo.get(lib, {}).get(metric)
        s5 = cov_500.get(lib, {}).get(metric)
        s10 = cov_1000.get(lib, {}).get(metric)
        d5 = diff_value(s5, t)
        d10 = diff_value(s10, t)

        row = (
            f"{project_cell(lib, marked_libs)} & {lp(t)} "
            f"& {lp(s5)} & {delta_cell(d5)} "
            f"& {lp(s10)} & {delta_cell(d10)} \\\\"
        )
        lines.append(row)
        if lib in SEPARATOR_LIBS:
            lines.append(r"\midrule")

        if t is not None: turbo_vals.append(t)
        if s5 is not None: s500_vals.append(s5)
        if s10 is not None: s1000_vals.append(s10)
        if d5 is not None: s500_deltas.append(d5)
        if d10 is not None: s1000_deltas.append(d10)

    lines.append(r"\midrule")
    med_t = _med(turbo_vals)
    med_5 = _med(s500_vals)
    med_10 = _med(s1000_vals)
    med_d5 = _med(s500_deltas)
    med_d10 = _med(s1000_deltas)
    lines.append(
        f"\\textbf{{Median score}} & {lp(med_t)} & {lp(med_5)} & -- "
        f"& {lp(med_10)} & -- \\\\"
    )
    lines.append(
        f"\\textbf{{Median $\\Delta$}} & -- & -- & {delta_cell(med_d5)} "
        f"& -- & {delta_cell(med_d10)} \\\\"
    )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    maybe_add_validation_note(lines, marked_libs)
    lines.append(r"\end{table*}")

    write_table(output_path, "\n".join(lines) + "\n")


def generate_testpilot_full_metrics_table(cov_turbo, cov_500, cov_1000,
                                          mut_turbo, mut_500, mut_1000, output_path):
    """Appendix reference table giving the exact numbers behind the comparison
    figure (fig:gpt5-testpilot-dumbbell): statement, branch, and line coverage
    plus mutation score, for GPT-Turbo and the two GPT-5.4 TestPilot budgets.

    All four metrics for all three models fit in one portrait table*: dropping the
    literal % from every cell (noted in the caption) keeps 12 numeric columns
    inside \\resizebox{\\textwidth}. Overleaf: testpilot/.
    """
    caption = (
        "Statement, branch, and line coverage and mutation score per package for "
        "GPT-Turbo and the two GPT-5.4 \\textsc{TestPilot} variants. These are the "
        "exact values used in Figure~\\ref{fig:gpt5-testpilot-dumbbell}. GPT-Turbo "
        "values are medians over 10 runs, and each GPT-5.4 value is from a single "
        "run. Rows are grouped by the original categories: Nessie packages, extra "
        "GitHub packages, and GitLab packages."
    )

    def nf(v):
        return f"{v:.1f}" if v is not None else "--"

    lines = []
    lines.append(r"%WARNING: auto-generated. Edit generate_gpt5_comparison_tables.py instead.")
    lines.append(r"\begin{table*}[!htb]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(r"\label{tab:gpt5-testpilot-full-metrics}")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{l rrrr rrrr rrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"\multirow{2}{*}{\textbf{Project}} & "
        r"\multicolumn{4}{c}{\textbf{GPT-Turbo}} & "
        r"\multicolumn{4}{c}{\textbf{GPT-5.4-500}} & "
        r"\multicolumn{4}{c}{\textbf{GPT-5.4-1000}} \\"
    )
    lines.append(r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}")
    hdr = r"\textbf{Stmt} & \textbf{Br} & \textbf{Line} & \textbf{Mut}"
    lines.append(f" & {hdr} & {hdr} & {hdr} \\\\")
    lines.append(r"\midrule")

    cols = {k: [] for k in (
        "t_stmt", "t_br", "t_line", "t_mut",
        "s5_stmt", "s5_br", "s5_line", "s5_mut",
        "s10_stmt", "s10_br", "s10_line", "s10_mut")}

    for lib in DOMAIN_MAP:
        if lib in COVERAGE_EXCLUDED_LIBS:
            continue
        vals = {
            "t_stmt":  cov_turbo.get(lib, {}).get("stmt_cov"),
            "t_br":    cov_turbo.get(lib, {}).get("branch_cov"),
            "t_line":  cov_turbo.get(lib, {}).get("line_cov"),
            "t_mut":   _mutation_score_or_zero_if_no_passing(mut_turbo, cov_turbo, lib),
            "s5_stmt":  cov_500.get(lib, {}).get("stmt_cov"),
            "s5_br":    cov_500.get(lib, {}).get("branch_cov"),
            "s5_line":  cov_500.get(lib, {}).get("line_cov"),
            "s5_mut":   _mutation_score_or_zero_if_no_passing(mut_500, cov_500, lib),
            "s10_stmt": cov_1000.get(lib, {}).get("stmt_cov"),
            "s10_br":   cov_1000.get(lib, {}).get("branch_cov"),
            "s10_line": cov_1000.get(lib, {}).get("line_cov"),
            "s10_mut":  _mutation_score_or_zero_if_no_passing(mut_1000, cov_1000, lib),
        }
        order = ("t_stmt", "t_br", "t_line", "t_mut",
                 "s5_stmt", "s5_br", "s5_line", "s5_mut",
                 "s10_stmt", "s10_br", "s10_line", "s10_mut")
        lines.append(f"{lib} & " + " & ".join(nf(vals[k]) for k in order) + r" \\")
        if lib in SEPARATOR_LIBS:
            lines.append(r"\midrule")
        for k in order:
            if vals[k] is not None:
                cols[k].append(vals[k])

    lines.append(r"\midrule")
    order = ("t_stmt", "t_br", "t_line", "t_mut",
             "s5_stmt", "s5_br", "s5_line", "s5_mut",
             "s10_stmt", "s10_br", "s10_line", "s10_mut")
    lines.append(r"\textbf{Median} & " + " & ".join(nf(_med(cols[k])) for k in order) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table*}")

    write_table(output_path, "\n".join(lines) + "\n")


def generate_mutation_table(cov_turbo, cov_500, cov_1000, cov_agent, mut_turbo, mut_500, mut_1000, mut_agent, output_path):
    caption = (
        "Mutation score comparison across GPT-Turbo, the two GPT-5.4 \\textsc{TestPilot} "
        "variants, and the GPT-5.4 agent. $\\Delta$ reports the absolute per-project "
        "difference in percentage points relative to GPT-Turbo. No-passing-test failures "
        "are counted as 0.0\\% mutation score. Row groups follow the original benchmark "
        "grouping: Nessie packages, additional GitHub packages, and GitLab packages."
    )

    lines = []
    lines.append(r"%WARNING: auto-generated. Edit generate_gpt5_comparison_tables.py instead.")
    lines.append(r"\begin{table*}[!htb]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(r"\label{tab:gpt5-mutation-comparison}")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{l r rr rr rr}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Project} & \textbf{GPT-Turbo} & "
        r"\multicolumn{2}{c}{\textbf{GPT-5.4-500}} & "
        r"\multicolumn{2}{c}{\textbf{GPT-5.4-1000}} & "
        r"\multicolumn{2}{c}{\textbf{GPT-5.4-agent}} \\"
    )
    lines.append(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}")
    lines.append(
        r"& \textbf{Mut.\ Score} "
        r"& \textbf{Mut.\ Score} & \textbf{$\Delta$ (pp)} "
        r"& \textbf{Mut.\ Score} & \textbf{$\Delta$ (pp)} "
        r"& \textbf{Mut.\ Score} & \textbf{$\Delta$ (pp)} \\"
    )
    lines.append(r"\midrule")

    mt_vals, m5_vals, m10_vals, mag_vals = [], [], [], []
    m5_deltas, m10_deltas, mag_deltas = [], [], []

    for lib in DOMAIN_MAP:
        if lib in NO_MUTATION_BASELINE_LIBS:
            continue

        mt  = _mutation_score_or_zero_if_no_passing(mut_turbo, cov_turbo, lib)
        m5  = _mutation_score_or_zero_if_no_passing(mut_500, cov_500, lib)
        m10 = _mutation_score_or_zero_if_no_passing(mut_1000, cov_1000, lib)
        mag = _mutation_score_or_zero_if_no_passing(mut_agent, cov_agent, lib)
        d5 = diff_value(m5, mt)
        d10 = diff_value(m10, mt)
        dag = diff_value(mag, mt)

        row = (
            f"{project_cell(lib, MUTATION_PROBLEM_LIBS)} & {lp(mt)} "
            f"& {lp(m5)} & {delta_cell(d5)} "
            f"& {lp(m10)} & {delta_cell(d10)} "
            f"& {lp(mag)} & {delta_cell(dag)} \\\\"
        )
        lines.append(row)
        if lib in SEPARATOR_LIBS:
            lines.append(r"\midrule")

        if mt  is not None: mt_vals.append(mt)
        if m5  is not None: m5_vals.append(m5)
        if m10 is not None: m10_vals.append(m10)
        if mag is not None: mag_vals.append(mag)
        if d5 is not None: m5_deltas.append(d5)
        if d10 is not None: m10_deltas.append(d10)
        if dag is not None: mag_deltas.append(dag)

    lines.append(r"\midrule")
    med_mt  = _med(mt_vals)
    med_m5  = _med(m5_vals)
    med_m10 = _med(m10_vals)
    med_mag = _med(mag_vals)
    med_d5 = _med(m5_deltas)
    med_d10 = _med(m10_deltas)
    med_dag = _med(mag_deltas)
    lines.append(
        f"\\textbf{{Median score}} & {lp(med_mt)} "
        f"& {lp(med_m5)} & -- "
        f"& {lp(med_m10)} & -- "
        f"& {lp(med_mag)} & -- \\\\"
    )
    lines.append(
        f"\\textbf{{Median $\\Delta$}} & -- "
        f"& -- & {delta_cell(med_d5)} "
        f"& -- & {delta_cell(med_d10)} "
        f"& -- & {delta_cell(med_dag)} \\\\"
    )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    maybe_add_validation_note(lines, MUTATION_PROBLEM_LIBS)
    lines.append(r"\end{table*}")

    write_table(output_path, "\n".join(lines) + "\n")


def generate_testpilot_mutation_table(cov_turbo, cov_500, cov_1000, mut_turbo, mut_500, mut_1000, output_path):
    caption = (
        "Mutation score comparison between GPT-Turbo, the strongest model in the original "
        "\\textsc{TestPilot} study, and GPT-5.4 running the same \\textsc{TestPilot} "
        "generation approach with two output-token budgets. $\\Delta$ reports the absolute "
        "per-project difference in percentage points relative to GPT-Turbo. No-passing-test "
        "failures are counted as 0.0\\% mutation score. Row groups follow the original "
        "benchmark grouping: Nessie packages, additional GitHub packages, and GitLab packages."
    )

    lines = []
    lines.append(r"%WARNING: auto-generated. Edit generate_gpt5_comparison_tables.py instead.")
    lines.append(r"\begin{table*}[!htb]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(r"\label{tab:gpt5-testpilot-mutation-comparison}")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{l r rr rr}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Project} & \textbf{GPT-Turbo} & "
        r"\multicolumn{2}{c}{\textbf{GPT-5.4-500}} & "
        r"\multicolumn{2}{c}{\textbf{GPT-5.4-1000}} \\"
    )
    lines.append(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}")
    lines.append(
        r"& \textbf{Mut.\ Score} "
        r"& \textbf{Mut.\ Score} & \textbf{$\Delta$ (pp)} "
        r"& \textbf{Mut.\ Score} & \textbf{$\Delta$ (pp)} \\"
    )
    lines.append(r"\midrule")

    mt_vals, m5_vals, m10_vals = [], [], []
    m5_deltas, m10_deltas = [], []

    for lib in DOMAIN_MAP:
        if lib in NO_MUTATION_BASELINE_LIBS:
            continue

        mt  = _mutation_score_or_zero_if_no_passing(mut_turbo, cov_turbo, lib)
        m5  = _mutation_score_or_zero_if_no_passing(mut_500, cov_500, lib)
        m10 = _mutation_score_or_zero_if_no_passing(mut_1000, cov_1000, lib)
        d5 = diff_value(m5, mt)
        d10 = diff_value(m10, mt)

        row = (
            f"{project_cell(lib, MUTATION_PROBLEM_LIBS)} & {lp(mt)} "
            f"& {lp(m5)} & {delta_cell(d5)} "
            f"& {lp(m10)} & {delta_cell(d10)} \\\\"
        )
        lines.append(row)
        if lib in SEPARATOR_LIBS:
            lines.append(r"\midrule")

        if mt  is not None: mt_vals.append(mt)
        if m5  is not None: m5_vals.append(m5)
        if m10 is not None: m10_vals.append(m10)
        if d5 is not None: m5_deltas.append(d5)
        if d10 is not None: m10_deltas.append(d10)

    lines.append(r"\midrule")
    med_mt  = _med(mt_vals)
    med_m5  = _med(m5_vals)
    med_m10 = _med(m10_vals)
    med_d5 = _med(m5_deltas)
    med_d10 = _med(m10_deltas)
    lines.append(
        f"\\textbf{{Median score}} & {lp(med_mt)} "
        f"& {lp(med_m5)} & -- "
        f"& {lp(med_m10)} & -- \\\\"
    )
    lines.append(
        f"\\textbf{{Median $\\Delta$}} & -- "
        f"& -- & {delta_cell(med_d5)} "
        f"& -- & {delta_cell(med_d10)} \\\\"
    )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    maybe_add_validation_note(lines, MUTATION_PROBLEM_LIBS)
    lines.append(r"\end{table*}")

    write_table(output_path, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# General mutation score table (one per GPT-5.4 variant)
# ---------------------------------------------------------------------------

def collect_loading_coverage(model_dir):
    """
    Read loading coverage from stats.json.
    Returns {lib: {stmt: float, branch: float}} for run "1".
    """
    result = {}
    run_dir = os.path.join(model_dir, "1")
    if not os.path.isdir(run_dir):
        return result
    for lib in os.listdir(run_dir):
        if lib.startswith("."):
            continue
        stats = read_json(os.path.join(run_dir, lib, "stats.json"))
        if stats is None:
            continue
        cfl = stats.get("coverageFromLoading", {})
        result[lib] = {
            "stmt":   cfl.get("statements", {}).get("pct"),
            "branch": cfl.get("branches",   {}).get("pct"),
        }
    return result


def lp_load(val):
    """Format loading-coverage percentage, or N/A when unavailable."""
    if val is None or val == "Unknown":
        return "N/A"
    try:
        return f"{float(val):.1f}\\%"
    except (TypeError, ValueError):
        return "N/A"


def generate_general_mutation_score_table(model_label, cov_data, mut_data,
                                          output_path, is_agent=False):
    """
    Produce a per-library mutation score table.

    cov_data  – {lib: {stmt_cov, branch_cov, passing, discarded}}
    mut_data  – {lib: score}
    is_agent  – when True, discarded column shows N/A (no dryRunStatus in agent reports)
    """
    if is_agent:
        caption = (
            f"Statement and branch coverage and mutation score for the {model_label} run."
        )
    else:
        caption = (
            f"Statement and branch coverage and mutation score for \\testpilot's passing tests, "
            f"generated using {model_label}. Discarded = tests that pass the test runner but "
            f"fail the mutation dry run. Libraries with no passing tests are shown as 0.0\\% "
            f"mutation score."
        )

    lines = []
    lines.append(r"%WARNING: auto-generated. Edit generate_gpt5_comparison_tables.py instead.")
    lines.append(r"\begin{table*}[!htb]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{tab:general-mutation-score-{model_label.lower().replace(' ', '-')}}}")
    lines.append(r"\resizebox{0.9\textwidth}{!}{")
    if is_agent:
        lines.append(r"\begin{tabular}{lrrrr}")
        lines.append(r"\toprule")
        lines.append(
            r"\textbf{Project} & \textbf{Passing Tests} & \textbf{Stmt Cov} & "
            r"\textbf{Branch Cov} & \textbf{Mutation Score} \\"
        )
    else:
        lines.append(r"\begin{tabular}{lrrrrrr}")
        lines.append(r"\toprule")
        lines.append(
            r"\textbf{Project} & \textbf{Passing Tests} & \textbf{Stmt Cov} & "
            r"\textbf{Branch Cov} & \textbf{Discarded} & \textbf{Mutation Score} \\"
        )
    lines.append(r"\midrule")

    for lib in DOMAIN_MAP:
        if lib in NO_MUTATION_BASELINE_LIBS:
            continue

        cov       = cov_data.get(lib, {})
        mut_score = _mutation_score_or_zero_if_no_passing(mut_data, cov_data, lib)

        stmt_cov   = cov.get("stmt_cov")
        branch_cov = cov.get("branch_cov")
        passing    = cov.get("passing")
        discarded  = cov.get("discarded")

        passing_fmt  = f"{round(passing)}"  if passing   is not None else "--"
        stmt_fmt     = f"{stmt_cov:.1f}\\%" if stmt_cov   is not None else "--"
        branch_fmt   = f"{branch_cov:.1f}\\%" if branch_cov is not None else "--"
        mut_fmt      = f"{mut_score:.1f}\\%" if mut_score  is not None else "--"
        if is_agent and not passing:
            # no-test libraries show no score in the pilot appendix table
            mut_fmt = "--"
        disc_fmt     = discard_cell(discarded, passing)

        if is_agent:
            lines.append(
                f"{lib} & {passing_fmt} & {stmt_fmt} & {branch_fmt} "
                f"& {mut_fmt} \\\\"
            )
        else:
            lines.append(
                f"{lib} & {passing_fmt} & {stmt_fmt} & {branch_fmt} "
                f"& {disc_fmt} & {mut_fmt} \\\\"
            )
        if lib in SEPARATOR_LIBS:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table*}")

    write_table(output_path, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading coverage stats...")
    cov_turbo = collect_coverage_stats(GPTTURBO_DIR)
    cov_500   = collect_coverage_stats(GPT54_500_DIR)
    cov_1000  = collect_coverage_stats(GPT54_1000_DIR)

    print("Loading coverage stats for agent...")
    cov_agent = collect_agent_coverage_stats(GPT54_AGENT_DIR)

    print("Loading mutation scores...")
    mut_turbo = collect_mutation_scores(GPTTURBO_DIR)
    mut_500   = collect_mutation_scores(GPT54_500_DIR)
    mut_1000  = collect_mutation_scores(GPT54_1000_DIR)
    mut_agent = collect_agent_mutation_scores(GPT54_AGENT_DIR, agent_cov_data=cov_agent)

    generate_cov_table(
        cov_turbo, cov_500, cov_1000,
        metric="stmt_cov",
        label="tab:gpt5-stmt-cov-comparison",
        caption=(
            "Statement coverage comparison between GPT-Turbo and the two GPT-5.4 "
            "\\textsc{TestPilot} variants. GPT-Turbo values are medians across 10 runs, "
            "while GPT-5.4 values are from a single run. $\\Delta$ reports the absolute "
            "per-project difference in percentage points relative to GPT-Turbo. Row groups "
            "follow the original benchmark grouping: Nessie packages, additional GitHub "
            "packages, and GitLab packages."
        ),
        output_path=os.path.join(TABLES_DIR, "tab-gpt5-stmt-cov-comparison.tex"),
        marked_libs=COVERAGE_PROBLEM_LIBS,
    )

    generate_cov_table(
        cov_turbo, cov_500, cov_1000,
        metric="branch_cov",
        label="tab:gpt5-branch-cov-comparison",
        caption=(
            "Branch coverage comparison between GPT-Turbo and the two GPT-5.4 "
            "\\textsc{TestPilot} variants. GPT-Turbo values are medians across 10 runs, "
            "while GPT-5.4 values are from a single run. $\\Delta$ reports the absolute "
            "per-project difference in percentage points relative to GPT-Turbo. Row groups "
            "follow the original benchmark grouping: Nessie packages, additional GitHub "
            "packages, and GitLab packages."
        ),
        output_path=os.path.join(TABLES_DIR, "tab-gpt5-branch-cov-comparison.tex"),
        marked_libs=COVERAGE_PROBLEM_LIBS,
    )

    generate_mutation_table(
        cov_turbo, cov_500, cov_1000, cov_agent,
        mut_turbo, mut_500, mut_1000, mut_agent,
        output_path=os.path.join(TABLES_DIR, "tab-gpt5-mutation-comparison.tex"),
    )
    generate_testpilot_mutation_table(
        cov_turbo, cov_500, cov_1000,
        mut_turbo, mut_500, mut_1000,
        output_path=os.path.join(TABLES_DIR, "tab-gpt5-testpilot-mutation-comparison.tex"),
    )
    generate_testpilot_full_metrics_table(
        cov_turbo, cov_500, cov_1000,
        mut_turbo, mut_500, mut_1000,
        output_path=os.path.join(TABLES_DIR, "tab-gpt5-testpilot-full-metrics.tex"),
    )

    print("Generating standalone mutation score tables...")
    generate_general_mutation_score_table(
        "GPT-5.4-500", cov_500, mut_500,
        os.path.join(TABLES_DIR, "tab-gpt5-500-general-mutation-score.tex"),
    )
    generate_general_mutation_score_table(
        "GPT-5.4-1000", cov_1000, mut_1000,
        os.path.join(TABLES_DIR, "tab-gpt5-1000-general-mutation-score.tex"),
    )
    generate_general_mutation_score_table(
        "GPT-5.4-agent", cov_agent, mut_agent,
        os.path.join(TABLES_DIR, "tab-gpt5-agent-general-mutation-score.tex"),
        is_agent=True,
    )

    print("Done.")


if __name__ == "__main__":
    main()
