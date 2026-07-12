#!/usr/bin/env python3
"""
Generate a LaTeX table comparing token usage between GPT-Turbo (estimated) and
GPT-5.4-output-500 / GPT-5.4-output-1000 / GPT-5.4-agent (actual counts).

GPT-Turbo estimation (upper bound, matches generate_token_table.py):
  - Input  = sum(promptLength // 4)  per query in codexQueryTimes.json
  - Output = sum(max_tokens * n)     per query in codexQueryTimes.json

GPT-5.4-output-500/1000 actual counts come from report.json > tokenUsage:
  - inputTokens, outputTokens

GPT-5.4-agent actual counts come from report/report.json > agentRun.llm:
  - inputTokens, outputTokens

Output (written flat to ./tables/; the thesis/Overleaf project sorts these into
subdirs):
  ./tables/tab-gpt5-token-comparison.tex            [Overleaf: pilot/]  — includes the agent column
  ./tables/tab-gpt5-testpilot-token-comparison.tex  [Overleaf: testpilot/] — no agent column (TestPilot + FUT only)

IMPORTANT — two output families, one script:
  * The testpilot/ table has NO agent column.
  * The pilot/ table's GPT-5.4-agent column is sourced from data/gpt-5.4-agent/ —
    the single OLD PILOT run that PREDATES the mutation-feedback/residual flags
    (a different, older agent commit than the 63be77f R3/R4 set used by the final
    result tables). These are the pilot numbers that motivated later agent
    changes, not the same-SHA result tables. This script is shared by both
    families, so it cannot be deleted as "pilot-only".
"""

import csv
import json
import os
import statistics

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, "data")
TABLES_DIR  = os.path.join(SCRIPT_DIR, "tables")

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

# Excluded thesis-wide for now (dropped from both token tables' rows and totals).
EXCLUDE_LIBS = {"fs-extra"}


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _avg(values):
    return round(sum(values) / len(values)) if values else None


def collect_gptturbo_tokens():
    """
    Estimate input/output tokens from codexQueryTimes.json (upper bound).
    Returns {lib: {input: int, output: int}} averaged across runs.
    """
    acc = {}
    for run in sorted(os.listdir(GPTTURBO_DIR)):
        if run.startswith("."):
            continue
        run_dir = os.path.join(GPTTURBO_DIR, run)
        if not os.path.isdir(run_dir):
            continue
        for lib in os.listdir(run_dir):
            if lib.startswith(".") or lib not in DOMAIN_MAP:
                continue
            token_file = os.path.join(run_dir, lib, "codexQueryTimes.json")
            queries = read_json(token_file)
            if queries is None:
                continue

            input_tok = sum(q[0].get("promptLength", 0) // 4 for q in queries)
            output_tok = sum(
                q[0].get("max_completion_tokens", q[0].get("max_tokens", 0)) * q[0].get("n", 1)
                for q in queries
            )

            entry = acc.setdefault(lib, {"input": [], "output": []})
            entry["input"].append(input_tok)
            entry["output"].append(output_tok)

    return {lib: {"input": _avg(v["input"]), "output": _avg(v["output"])} for lib, v in acc.items()}


def collect_gpt5_tokens(model_dir):
    """
    Read actual token counts from report.json > tokenUsage.
    Returns {lib: {input: int, output: int}} (single run, run ID "1").
    """
    result = {}
    run = "1"
    run_dir = os.path.join(model_dir, run)
    if not os.path.isdir(run_dir):
        return result

    for lib in os.listdir(run_dir):
        if lib.startswith(".") or lib not in DOMAIN_MAP:
            continue
        report = read_json(os.path.join(run_dir, lib, "report.json"))
        if report is None:
            continue
        usage = report.get("tokenUsage")
        if usage is None:
            continue
        result[lib] = {
            "input":  usage.get("inputTokens"),
            "output": usage.get("outputTokens"),
        }

    return result


def collect_gpt5_agent_tokens(model_dir):
    """
    Read actual token counts from report/report.json > agentRun.llm.
    Returns {lib: {input: int, output: int}} (single run, run ID "1").
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
        llm = report.get("agentRun", {}).get("llm")
        if llm is None:
            continue
        result[lib] = {
            "input":  llm.get("inputTokens"),
            "output": llm.get("outputTokens"),
        }

    return result


def collect_function_counts():
    """
    Count unique API functions targeted by TestPilot for each library.
    Uses GPT-5.4-1000 report.json as the primary source (most complete run),
    falling back to GPT-5.4-500 and then GPT-Turbo.
    """
    result = {}
    sources = [
        os.path.join(GPT54_1000_DIR, "1"),
        os.path.join(GPT54_500_DIR, "1"),
    ]
    # Add all turbo runs as fallback
    if os.path.isdir(GPTTURBO_DIR):
        for run in sorted(os.listdir(GPTTURBO_DIR)):
            p = os.path.join(GPTTURBO_DIR, run)
            if os.path.isdir(p):
                sources.append(p)

    for run_dir in sources:
        if not os.path.isdir(run_dir):
            continue
        for lib in os.listdir(run_dir):
            if lib in result or lib.startswith(".") or lib not in DOMAIN_MAP:
                continue
            report = read_json(os.path.join(run_dir, lib, "report.json"))
            if report is None:
                continue
            tests = report.get("tests", [])
            apis = set()
            for t in tests:
                api = t.get("api")
                if api:
                    apis.add(api)
            if apis:
                result[lib] = len(apis)
    return result


def collect_loc():
    """
    Read LOC from stats.json for each library. LOC is a property of the library,
    not the model run, so we read from whichever model has the data (gpt-5.4-output-1000).
    Falls back to gptturbo if a library is missing from gpt-5.4-output-1000.
    """
    result = {}
    for model_run in [
        os.path.join(GPT54_1000_DIR, "1"),
        os.path.join(GPTTURBO_DIR, sorted(os.listdir(GPTTURBO_DIR))[0]),
    ]:
        if not os.path.isdir(model_run):
            continue
        for lib in os.listdir(model_run):
            if lib in result:
                continue
            stats = read_json(os.path.join(model_run, lib, "stats.json"))
            if stats and stats.get("loc") is not None:
                result[lib] = stats["loc"]
    return result


def fmt(val):
    if val is None:
        return "---"
    rounded = round(val / 1000) * 1000
    if rounded >= 1_000_000:
        m = rounded / 1_000_000
        return f"{m:.3f}".rstrip("0").rstrip(".") + "M"
    return f"{rounded // 1000}k"


def main():
    print("Collecting token data...")
    turbo  = collect_gptturbo_tokens()
    s500   = collect_gpt5_tokens(GPT54_500_DIR)
    s1000  = collect_gpt5_tokens(GPT54_1000_DIR)
    agent  = collect_gpt5_agent_tokens(GPT54_AGENT_DIR)
    loc    = collect_loc()
    fut    = collect_function_counts()

    totals = {
        "turbo":  {"input": 0, "output": 0},
        "s500":   {"input": 0, "output": 0},
        "s1000":  {"input": 0, "output": 0},
        "agent":  {"input": 0, "output": 0},
        "loc":    0,
        "fut":    0,
    }

    rows_full = ""       # for tab-gpt5-token-comparison (includes agent)
    rows_testpilot = ""  # for tab-gpt5-testpilot-token-comparison (no agent, has FUT)
    for lib in DOMAIN_MAP:
        if lib in EXCLUDE_LIBS:
            continue
        t   = turbo.get(lib, {})
        g5  = s500.get(lib, {})
        g10 = s1000.get(lib, {})
        ag  = agent.get(lib, {})
        l   = loc.get(lib)
        f   = fut.get(lib)

        ti,  to   = t.get("input"),   t.get("output")
        g5i, g5o  = g5.get("input"),  g5.get("output")
        g10i, g10o = g10.get("input"), g10.get("output")
        agi, ago  = ag.get("input"),  ag.get("output")

        loc_fmt = str(l) if l is not None else "---"
        fut_fmt = str(f) if f is not None else "---"
        rows_full += (
            f"{lib} & {loc_fmt} & {fut_fmt} "
            f"& {fmt(g5i)} & {fmt(g5o)} "
            f"& {fmt(g10i)} & {fmt(g10o)} "
            f"& {fmt(agi)} & {fmt(ago)} \\\\\n"
        )
        rows_testpilot += (
            f"{lib} & {loc_fmt} & {fut_fmt} & {fmt(ti)} & {fmt(to)} "
            f"& {fmt(g5i)} & {fmt(g5o)} "
            f"& {fmt(g10i)} & {fmt(g10o)} \\\\\n"
        )
        if lib in SEPARATOR_LIBS:
            rows_full += "\\midrule\n"
            rows_testpilot += "\\midrule\n"

        if l is not None:
            totals["loc"] += l
        if f is not None:
            totals["fut"] += f
        for key, val in [("input", ti), ("output", to)]:
            if val is not None:
                totals["turbo"][key] += val
        for key, val in [("input", g5i), ("output", g5o)]:
            if val is not None:
                totals["s500"][key] += val
        for key, val in [("input", g10i), ("output", g10o)]:
            if val is not None:
                totals["s1000"][key] += val
        for key, val in [("input", agi), ("output", ago)]:
            if val is not None:
                totals["agent"][key] += val

    rows_full += (
        "\\midrule\n"
        f"\\textbf{{Total}} & {totals['loc']} & {totals['fut']} "
        f"& {fmt(totals['s500']['input'])} & {fmt(totals['s500']['output'])} "
        f"& {fmt(totals['s1000']['input'])} & {fmt(totals['s1000']['output'])} "
        f"& {fmt(totals['agent']['input'])} & {fmt(totals['agent']['output'])} \\\\\n"
    )
    rows_testpilot += (
        "\\midrule\n"
        f"\\textbf{{Total}} & {totals['loc']} & {totals['fut']} "
        f"& {fmt(totals['turbo']['input'])} & {fmt(totals['turbo']['output'])} "
        f"& {fmt(totals['s500']['input'])} & {fmt(totals['s500']['output'])} "
        f"& {fmt(totals['s1000']['input'])} & {fmt(totals['s1000']['output'])} \\\\\n"
    )

    # --- Full table (GPT-5.4 TestPilot + agent, with FUT column) ---
    table_full = r"""%WARNING: auto-generated. Edit generate_gpt5_token_table.py instead.
\begin{table*}[!htb]
\centering
\caption{Token usage comparison between GPT-5.4 TestPilot and the GPT-5.4 agent.
  FUT is the number of exported functions under test.
  All values are actual counts from the API.}
\label{tab:gpt5-token-comparison}
\resizebox{\textwidth}{!}{
\begin{tabular}{l r r rr rr rr}
\toprule
& & & \multicolumn{2}{c}{\textbf{GPT-5.4-500}} & \multicolumn{2}{c}{\textbf{GPT-5.4-1000}} & \multicolumn{2}{c}{\textbf{GPT-5.4-agent}} \\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
\textbf{Project} & \textbf{LoC} & \textbf{FUT} & \textbf{Input} & \textbf{Output} & \textbf{Input} & \textbf{Output} & \textbf{Input} & \textbf{Output} \\
\midrule
""" + rows_full + r"""\bottomrule
\end{tabular}
}
\end{table*}
"""

    # --- TestPilot-only table (with FUT column, no agent) ---
    table_testpilot = r"""%WARNING: auto-generated. Edit generate_gpt5_token_table.py instead.
\begin{table*}[!htb]
\centering
\caption{Token usage for GPT-Turbo and GPT-5.4 running TestPilot's generation approach.
  FUT is the number of exported functions under test.
  GPT-Turbo values are estimated upper bounds (input tokens estimated as prompt length
  divided by four; output tokens as the output token limit times number of completions).
  GPT-5.4 values are actual counts from the API.}
\label{tab:gpt5-testpilot-token-comparison}
\resizebox{\textwidth}{!}{
\begin{tabular}{l r r rr rr rr}
\toprule
& & & \multicolumn{2}{c}{\textbf{GPT-Turbo} (est.)} & \multicolumn{2}{c}{\textbf{GPT-5.4-500} (actual)} & \multicolumn{2}{c}{\textbf{GPT-5.4-1000} (actual)} \\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
\textbf{Project} & \textbf{LoC} & \textbf{FUT} & \textbf{Input} & \textbf{Output} & \textbf{Input} & \textbf{Output} & \textbf{Input} & \textbf{Output} \\
\midrule
""" + rows_testpilot + r"""\bottomrule
\end{tabular}
}
\end{table*}
"""

    os.makedirs(TABLES_DIR, exist_ok=True)

    out_full = os.path.join(TABLES_DIR, "tab-gpt5-token-comparison.tex")
    with open(out_full, "w") as fh:
        fh.write(table_full)
    print(f"Written: {out_full}")

    out_tp = os.path.join(TABLES_DIR, "tab-gpt5-testpilot-token-comparison.tex")
    with open(out_tp, "w") as fh:
        fh.write(table_testpilot)
    print(f"Written: {out_tp}")

    # Raw (unrounded) per-library values, consumed by generate_gpt5_token_figure.py
    # so the figure can never drift from the tables above.
    csv_path = os.path.join(TABLES_DIR, "tab-gpt5-token-usage.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "library", "loc", "fut",
            "turbo_input", "turbo_output",
            "s500_input", "s500_output",
            "s1000_input", "s1000_output",
            "agent_input", "agent_output",
        ])
        for lib in DOMAIN_MAP:
            if lib in EXCLUDE_LIBS:
                continue
            t, g5, g10, ag = (turbo.get(lib, {}), s500.get(lib, {}),
                              s1000.get(lib, {}), agent.get(lib, {}))
            w.writerow([
                lib, loc.get(lib), fut.get(lib),
                t.get("input"), t.get("output"),
                g5.get("input"), g5.get("output"),
                g10.get("input"), g10.get("output"),
                ag.get("input"), ag.get("output"),
            ])
    print(f"Written: {csv_path}")


if __name__ == "__main__":
    main()
