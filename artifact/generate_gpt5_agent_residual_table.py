#!/usr/bin/env python3
"""
Generate evaluation tables for the GPT-5.4 agent residual-mutation-feedback runs.

Reads the two clean HEAD (agent sha 63be77f, temperature 1) full+residual replicates:
  data/gpt-5.4-agent-residual/head-63be77f/residual-21-gpt54-dryrun-repair-20260526/  (R3)
  data/gpt-5.4-agent-residual/head-63be77f/residual-21-gpt54-temp1-rep2-20260530/      (R4)

Both runs share the condition mutationFeedback=full, strykerDryRunRepair=on,
residual feedback enabled. RUN_IDS are "3"/"4" (the runs are R3/R4 in the
thesis run inventory); RUN_DIRS maps each logical id to its on-disk dir. The
older-SHA runs R1 (1-7064a0c) and R2 (2-cacfda3) are no longer consumed here;
they remain on disk for provenance and are analysed via
agentic-stryker/scripts/compareConditions.cjs.

Per-library score and cost values are means across the runs that produced data
for the cell. No-test generation outcomes are scored as 0.0% because the agent
produced no usable test suite. Other incomplete measurement runs are omitted
from score cells. Score/cost cells where only one run contributed are marked
with a trailing asterisk in the LaTeX output. Missing data is shown as an
em-dash ("--").

Outputs:
  tables/tab-gpt5-agent-residual-cost.tex      — token usage, cost, requests, time
  tables/tab-gpt5-agent-residual.csv           — flat per-run dump for inspection

tab-gpt5-agent-residual-scores.tex is NOT written here: it is owned by
generate_gpt5_agent_temperature_table.py, which extends it with temperature-0
columns (residual effect at both temperatures) under the same table label.

Pricing rates come from ./pricing.json. Verify those before believing the cost column.
The manually-written manifest.json files are NOT consulted.
"""

import csv
import json
import os
import sys

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(SCRIPT_DIR, "data", "gpt-5.4-agent-residual")
TABLES_DIR   = os.path.join(SCRIPT_DIR, "tables")
PRICING_FILE = os.path.join(SCRIPT_DIR, "pricing.json")

RUN_IDS = ["3", "4"]
# Logical run id -> on-disk dir. These are the HEAD (sha 63be77f) replicates
# R3/R4; the dir names carry full provenance.
RUN_DIRS = {
    "3": "head-63be77f/residual-21-gpt54-dryrun-repair-20260526",
    "4": "head-63be77f/residual-21-gpt54-temp1-rep2-20260530",
}

# Reused from generate_gpt5_token_table.py for visual alignment with existing tables.
DOMAIN_MAP = [
    "glob", "fs-extra", "graceful-fs", "jsonfile",
    "q", "node-dir", "zip-a-folder",
    "quill-delta", "complex.js", "pull-stream",
    "countries-and-timezones", "simple-statistics", "plural", "dirty",
    "geo-point", "uneval", "image-downloader", "crawler-url-parser",
    "gitlab-js", "core", "omnitool",
]

SEPARATOR_LIBS = {"zip-a-folder", "uneval"}

# Libraries dropped from the thesis-facing tables (excluded thesis-wide for now).
# Data is still collected so the stdout diagnostics see the full set; only the
# emitted table rows and the table totals skip these.
EXCLUDE_LIBS = {"fs-extra"}


# ---------- I/O helpers ----------

def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def load_pricing():
    data = read_json(PRICING_FILE)
    if data is None:
        sys.exit(f"missing pricing file: {PRICING_FILE}")
    return data["rates_per_million_tokens"]


# ---------- collection ----------

# Historical note: an omnitool re-evaluation splice (fed by
# agentic-omnitool-reeval/summary.json) used to patch the omnitool score cells
# of the older-SHA runs R1/R2. Those runs are no longer consumed here
# (RUN_IDS is "3"/"4"), so the splice was removed on 2026-07-03; see git
# history / agentic-omnitool-reeval/ for the archived re-evaluation data.

def collect_run(run_id):
    """Return {lib: per-cell dict} for one run."""
    run_dir = os.path.join(DATA_DIR, RUN_DIRS[run_id])
    suite = read_json(os.path.join(run_dir, "suite-report.json")) or {}
    suite_status = {l["libraryId"]: l.get("status") for l in suite.get("libraries", [])}

    out = {}
    for lib in DOMAIN_MAP:
        report = read_json(os.path.join(run_dir, lib, "report", "report.json"))
        if report is None:
            if lib in suite_status:
                out[lib] = {
                    "status":              suite_status.get(lib),
                    "duration_ms":         None,
                    "input_tokens":        None,
                    "cached_input_tokens": None,
                    "output_tokens":       None,
                    "reasoning_tokens":    None,
                    "requests":            None,
                    "tests_written":       None,
                    "tests_passing":       None,
                    "tests_failures":      None,
                    "tests_total":         None,
                    "fatal_error":         None,
                    "score_measurement_complete": False,
                    "score_splices":        [],
                    "pre_score":           None,
                    "residual_score":      None,
                    "total_valid":         None,
                    "benchmark_score":     None,
                    "benchmark_total_valid": None,
                    "benchmark_undetected": None,
                    "benchmark_exit_code": None,
                    "benchmark_delta":     None,
                    "benchmark_minus_agent_score": None,
                }
            continue
        ar    = report.get("agentRun", {})
        gen   = ar.get("generation", {})
        llm   = ar.get("llm", {})
        stats = report.get("stats", {})
        pre   = gen.get("preResidualAggregatePass") or {}
        agg   = gen.get("aggregatePass") or {}
        stryker = report.get("stryker") or {}
        stryker_metrics = stryker.get("metrics") or {}

        pre_score = pre.get("mutationScore") if pre.get("succeeded") else None
        residual_score = agg.get("mutationScore") if agg.get("succeeded") else None
        benchmark_score = stryker_metrics.get("mutationScore") if stryker_metrics else None

        cell = {
            "status":              suite_status.get(lib),
            "duration_ms":         ar.get("durationMs"),
            "input_tokens":        llm.get("inputTokens"),
            "cached_input_tokens": llm.get("cacheReadInputTokens"),
            "output_tokens":       llm.get("outputTokens"),
            "reasoning_tokens":    llm.get("reasoningTokens", 0),
            "requests":            llm.get("requests"),
            "tests_written":       gen.get("testsWritten"),
            "tests_passing":       stats.get("nrPasses"),
            "tests_failures":      stats.get("nrFailures"),
            "tests_total":         stats.get("nrTests"),
            "fatal_error":         ar.get("fatalError"),
            "pre_score":           pre_score,
            "residual_score":      residual_score,
            "total_valid":         (agg.get("totalValid") if agg.get("succeeded")
                                    else pre.get("totalValid") if pre.get("succeeded")
                                    else None),
            "benchmark_score":     benchmark_score,
            "benchmark_total_valid": (stryker_metrics.get("totalValid")
                                      if stryker_metrics else None),
            "benchmark_undetected": (stryker_metrics.get("totalUndetected")
                                     if stryker_metrics else None),
            "benchmark_exit_code": stryker.get("exitCode"),
            "benchmark_delta":     (benchmark_score - pre_score
                                    if benchmark_score is not None and pre_score is not None
                                    else None),
            "benchmark_minus_agent_score": (benchmark_score - residual_score
                                            if benchmark_score is not None
                                            and residual_score is not None
                                            else None),
        }
        out[lib] = finalize_score_cell(cell)
    return out


def score_measurement_complete(c):
    """True iff this run can contribute to the score columns."""
    if c is None:
        return False
    if c.get("status") in {"failed_benchmark", "failed_setup", "skipped"}:
        return False
    if c.get("fatal_error"):
        return False
    if (c.get("tests_failures") or 0) > 0:
        return False
    benchmark_exit_code = c.get("benchmark_exit_code")
    if benchmark_exit_code is not None and benchmark_exit_code != 0:
        return False
    return (
        c.get("pre_score") is not None
        and c.get("residual_score") is not None
        and c.get("benchmark_score") is not None
    )


def no_test_score_outcome(c):
    """True iff a failed run should still contribute a 0.0% score.

    This represents an end-to-end generation failure, not a measurement failure:
    the agent retained no tests, so the generated suite kills no mutants.
    """
    if c is None:
        return False
    if c.get("fatal_error"):
        return False
    if c.get("status") in {"failed_benchmark", "failed_setup", "skipped"}:
        return False
    return (
        c.get("tests_written") == 0
        and c.get("tests_total") == 0
        and (c.get("tests_failures") or 0) == 0
    )


def finalize_score_cell(c):
    """Keep status/test/cost data, but drop incomplete runs from score cells."""
    if c is None:
        return None
    complete = score_measurement_complete(c)
    c["score_measurement_complete"] = complete
    c["no_test_score"] = no_test_score_outcome(c)
    if complete:
        return c
    if c["no_test_score"]:
        c["pre_score"] = 0.0
        c["residual_score"] = 0.0
        c["benchmark_score"] = 0.0
        c["benchmark_delta"] = 0.0
        c["benchmark_minus_agent_score"] = 0.0
        c["score_splices"] = list(c.get("score_splices") or []) + ["no-test-zero-score"]
        return c
    for key in [
        "pre_score",
        "residual_score",
        "total_valid",
        "benchmark_score",
        "benchmark_total_valid",
        "benchmark_undetected",
        "benchmark_delta",
        "benchmark_minus_agent_score",
    ]:
        c[key] = None
    return c


def cell_cost(c, rates):
    if c is None:
        return None
    if all(c.get(k) is None for k in ["input_tokens", "cached_input_tokens",
                                      "output_tokens", "reasoning_tokens"]):
        return None
    inp     = c["input_tokens"]        or 0
    cached  = c["cached_input_tokens"] or 0
    output  = c["output_tokens"]       or 0
    reason  = c["reasoning_tokens"]    or 0
    non_cached = max(inp - cached, 0)
    cost = (
        non_cached * rates["input"]
        + cached    * rates["cached_input"]
        + output    * rates["output"]
        + reason    * rates.get("reasoning", rates["output"])
    ) / 1_000_000.0
    return cost


def display_status(c):
    """Collapsed status for the score table."""
    if c is None:
        return "--"
    if c.get("status") == "skipped":
        return "--"
    return "ok" if c.get("score_measurement_complete") else "fail"


def avg(values):
    """Mean of values, dropping Nones. Returns (mean, n)."""
    clean = [v for v in values if v is not None]
    if not clean:
        return None, 0
    return sum(clean) / len(clean), len(clean)


def aggregate(runs, rates):
    rows = {}
    for lib in DOMAIN_MAP:
        cells = [runs[r].get(lib) for r in RUN_IDS]
        present = [c for c in cells if c is not None]

        row = {"_n_runs_present": len(present)}
        for k in ["duration_ms", "input_tokens", "cached_input_tokens",
                  "output_tokens", "reasoning_tokens", "requests",
                  "tests_written", "tests_passing", "tests_failures", "tests_total",
                  "pre_score", "residual_score", "total_valid",
                  "benchmark_score", "benchmark_total_valid",
                  "benchmark_undetected", "benchmark_delta",
                  "benchmark_minus_agent_score"]:
            mean, n = avg([c[k] if c else None for c in cells])
            row[k] = mean
            row[k + "_n"] = n

        # Cost: per-run, then mean.
        row["cost"], row["cost_n"] = avg([cell_cost(c, rates) for c in present])

        # Delta = residual - pre per run, then mean.
        deltas = []
        for c in present:
            if c["pre_score"] is not None and c["residual_score"] is not None:
                deltas.append(c["residual_score"] - c["pre_score"])
        row["delta"], row["delta_n"] = avg(deltas)

        statuses = [display_status(c) for c in cells]
        row["status_pair"] = "/".join(statuses)

        rows[lib] = row
    return rows


# ---------- formatting ----------

N_RUNS = len(RUN_IDS)
STAR = r"\textsuperscript{*}"


def _star(n):
    return STAR if 0 < n < N_RUNS else ""


def fmt_token(v, n):
    if v is None:
        return "--"
    v = round(v)
    if   v >= 1_000_000: s = f"{v/1_000_000:.2f}M"
    elif v >= 1_000:     s = f"{v/1_000:.0f}k"
    else:                s = f"{v}"
    return s + _star(n)


def fmt_int(v, n):
    if v is None:
        return "--"
    return f"{round(v):,}" + _star(n)


def fmt_int_plain(v):
    if v is None:
        return "--"
    return f"{round(v):,}"


def fmt_tests_run(c):
    if c is None:
        return "--"
    passing = c.get("tests_passing")
    written = c.get("tests_written")
    if passing is None and written is None:
        return "--"
    return f"{fmt_int_plain(passing)}/{fmt_int_plain(written)}"


def fmt_float(v, n, ndigits=2):
    if v is None:
        return "--"
    return f"{v:.{ndigits}f}" + _star(n)


def fmt_pct(v, n):
    if v is None:
        return "--"
    return f"{v:.1f}\\%" + _star(n)


def fmt_minutes(ms, n):
    if ms is None:
        return "--"
    return f"{ms/60000:.1f}" + _star(n)


def fmt_delta(v, n):
    if v is None:
        return "--"
    sign = "+" if v >= 0 else "-"
    return f"${sign}${abs(v):.1f}" + _star(n)


_STATUS_SHORT = {
    "ok": "ok",
    "fail": "fail",
    "--": "--",
}


def fmt_status_pair(s):
    return "/".join(_STATUS_SHORT.get(p, p) for p in s.split("/"))


# ---------- writers ----------

WARN = "%WARNING: auto-generated. Edit generate_gpt5_agent_residual_table.py instead.\n"


def _separator_after(lib):
    return lib in SEPARATOR_LIBS


def write_cost_table(rows, path):
    lines = [
        WARN.rstrip("\n"),
        r"\begin{table*}[t!]",
        r"\centering",
        r"\caption{GPT-5.4 agent: token usage, cost, and generation time, "
        r"averaged across the two residual-mutation-feedback runs. "
        r"Cells marked \textsuperscript{*} come from a single run.}",
        r"\label{tab:gpt5-agent-residual-cost}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"\textbf{Project} & \textbf{Input} & \textbf{Cached} & \textbf{Output} & "
        r"\textbf{Cost (USD)} & \textbf{Requests} & \textbf{Time (min)} \\",
        r"\midrule",
    ]
    totals_keys = ["input_tokens", "cached_input_tokens", "output_tokens",
                   "cost", "requests", "duration_ms"]
    totals = {k: 0.0 for k in totals_keys}
    for lib in DOMAIN_MAP:
        if lib in EXCLUDE_LIBS:
            continue
        r = rows[lib]
        if r["_n_runs_present"] == 0:
            cells = ["--"] * 6
        else:
            cells = [
                fmt_token(r["input_tokens"],        r["input_tokens_n"]),
                fmt_token(r["cached_input_tokens"], r["cached_input_tokens_n"]),
                fmt_token(r["output_tokens"],       r["output_tokens_n"]),
                fmt_float(r["cost"],                r["cost_n"]),
                fmt_int(r["requests"],              r["requests_n"]),
                fmt_minutes(r["duration_ms"],       r["duration_ms_n"]),
            ]
            for k in totals_keys:
                if r.get(k) is not None:
                    totals[k] += r[k]
        lines.append(f"{lib} & " + " & ".join(cells) + r" \\")
        if _separator_after(lib):
            lines.append(r"\midrule")
    lines.append(r"\midrule")
    lines.append(
        r"\textbf{Total} & "
        + fmt_token(totals["input_tokens"],        N_RUNS) + " & "
        + fmt_token(totals["cached_input_tokens"], N_RUNS) + " & "
        + fmt_token(totals["output_tokens"],       N_RUNS) + " & "
        + fmt_float(totals["cost"],                N_RUNS) + " & "
        + fmt_int(totals["requests"],              N_RUNS) + " & "
        + fmt_minutes(totals["duration_ms"],       N_RUNS) + r" \\"
    )
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def write_csv(rows, runs, rates, path):
    fields = ["library"]
    metrics = [
        ("status",              False),
        ("display_status",      False),
        ("score_measurement_complete", False),
        ("no_test_score",       False),
        ("score_splices",       False),
        ("input_tokens",        True),
        ("cached_input_tokens", True),
        ("output_tokens",       True),
        ("reasoning_tokens",    True),
        ("cost_usd",            True),
        ("requests",            True),
        ("duration_ms",         True),
        ("tests_passing",       True),
        ("tests_failures",      True),
        ("tests_written",       True),
        ("tests_total",         True),
        ("total_valid",         True),
        ("pre_score",           True),
        ("residual_score",      True),
        ("benchmark_score",     True),
        ("benchmark_total_valid", True),
        ("benchmark_undetected", True),
        ("benchmark_exit_code", False),
        ("delta",               True),
        ("benchmark_delta",     True),
        ("benchmark_minus_agent_score", True),
    ]
    for name, has_mean in metrics:
        for r_id in RUN_IDS:
            fields.append(f"{name}_r{r_id}")
        if has_mean:
            fields.append(f"{name}_mean")
            fields.append(f"{name}_n")

    def per_run_value(c, key):
        if not c: return None
        if key == "display_status": return display_status(c)
        if key == "cost_usd": return cell_cost(c, rates)
        if key == "delta":
            if c["pre_score"] is None or c["residual_score"] is None: return None
            return c["residual_score"] - c["pre_score"]
        return c.get(key)

    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for lib in DOMAIN_MAP:
            row = {"library": lib}
            agg = rows[lib]
            for name, has_mean in metrics:
                for r_id in RUN_IDS:
                    row[f"{name}_r{r_id}"] = per_run_value(runs[r_id].get(lib), name)
                if has_mean:
                    if name == "cost_usd":
                        row[f"{name}_mean"] = agg["cost"]
                        row[f"{name}_n"]    = agg["cost_n"]
                    else:
                        row[f"{name}_mean"] = agg[name]
                        row[f"{name}_n"]    = agg[f"{name}_n"]
            w.writerow(row)


# ---------- entry point ----------

def main():
    rates = load_pricing()
    runs = {r: collect_run(r) for r in RUN_IDS}
    rows = aggregate(runs, rates)

    os.makedirs(TABLES_DIR, exist_ok=True)
    cost_path   = os.path.join(TABLES_DIR, "tab-gpt5-agent-residual-cost.tex")
    csv_path    = os.path.join(TABLES_DIR, "tab-gpt5-agent-residual.csv")
    write_cost_table(rows, cost_path)
    # tab-gpt5-agent-residual-scores.tex is produced by
    # generate_gpt5_agent_temperature_table.py (adds the temperature-0 columns).
    write_csv(rows, runs, rates, csv_path)

    libs_with_pre   = [l for l in DOMAIN_MAP if rows[l]["pre_score"]      is not None]
    libs_with_resi  = [l for l in DOMAIN_MAP if rows[l]["residual_score"] is not None]
    libs_with_bench = [l for l in DOMAIN_MAP if rows[l]["benchmark_score"] is not None]
    libs_with_delta = [l for l in DOMAIN_MAP if rows[l]["delta"]          is not None]
    discrepancies = []
    for r_id in RUN_IDS:
        for lib in DOMAIN_MAP:
            c = runs[r_id].get(lib)
            if not c:
                continue
            diff = c.get("benchmark_minus_agent_score")
            if diff is not None and abs(diff) > 0.10:
                discrepancies.append((r_id, lib, diff))

    def umean(vals):
        return (sum(vals) / len(vals)) if vals else 0.0

    print(f"wrote {os.path.relpath(cost_path,   SCRIPT_DIR)}")
    print(f"wrote {os.path.relpath(csv_path,    SCRIPT_DIR)}")
    print()
    print(f"libraries with usable pre-residual score : {len(libs_with_pre)}/{len(DOMAIN_MAP)}")
    print(f"libraries with usable residual score     : {len(libs_with_resi)}/{len(DOMAIN_MAP)}")
    print(f"libraries with usable benchmark score    : {len(libs_with_bench)}/{len(DOMAIN_MAP)}")
    print(f"libraries with usable Delta              : {len(libs_with_delta)}/{len(DOMAIN_MAP)}")
    print(f"unweighted mean pre-residual score (%)   : "
          f"{umean([rows[l]['pre_score']      for l in libs_with_pre]):.2f}")
    print(f"unweighted mean residual score (%)       : "
          f"{umean([rows[l]['residual_score'] for l in libs_with_resi]):.2f}")
    print(f"unweighted mean benchmark score (%)      : "
          f"{umean([rows[l]['benchmark_score'] for l in libs_with_bench]):.2f}")
    print(f"unweighted mean Delta (pp)               : "
          f"{umean([rows[l]['delta']          for l in libs_with_delta]):.2f}")
    if discrepancies:
        print("agent final vs benchmark score differences > 0.10 pp:")
        for r_id, lib, diff in discrepancies:
            print(f"  R{r_id} {lib}: {diff:+.2f} pp")
    total_cost = sum((rows[l]["cost"] or 0) for l in DOMAIN_MAP)
    print(f"total cost (sum of per-lib means, USD)   : ${total_cost:.2f}")


if __name__ == "__main__":
    main()
