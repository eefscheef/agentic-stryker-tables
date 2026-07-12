#!/usr/bin/env python3
"""
Generate the correctness failure overview table for GPT-5.4 TestPilot and agent runs.

The table uses a library/project-run denominator:
  - TestPilot rows count one run over the 21 benchmark libraries.
  - The agent row counts five final runs over the same 21 libraries.

Outputs:
  tables/tab-gpt5-correctness-failure-overview.tex
  tables/tab-gpt5-correctness-failure-overview.csv
"""

import csv
import json
import os
import sys

sys.dont_write_bytecode = True

import generate_gpt5_agent_nontriviality as agent_nontriviality
import generate_gpt5_comparison_tables as comparison

# fs-extra is excluded thesis-wide; mirror the comparison-table exclusion so the
# denominators and per-library loops drop it consistently.
LIBS = [lib for lib in comparison.DOMAIN_MAP
        if lib not in comparison.COVERAGE_EXCLUDED_LIBS]


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR = os.path.join(SCRIPT_DIR, "tables")
WARN = "%WARNING: auto-generated. Edit generate_gpt5_correctness_failure_table.py instead."

OUTPUT_TEX = os.path.join(TABLES_DIR, "tab-gpt5-correctness-failure-overview.tex")
OUTPUT_CSV = os.path.join(TABLES_DIR, "tab-gpt5-correctness-failure-overview.csv")

TESTPILOT_RUNS = [
    ("TestPilot-500", os.path.join(SCRIPT_DIR, "data", "gpt-5.4-output-500", "1")),
    ("TestPilot-1000", os.path.join(SCRIPT_DIR, "data", "gpt-5.4-output-1000", "1")),
]

# 2026-06-27: TestPilot-500/1000 counts dropped by one "outright" each after the
# fixed-harness crawler-url-parser splice (it now retains tests). fs-extra is also
# excluded thesis-wide, lowering each denominator by one library per run (21->20 for
# single-run systems, 42->40 for the 2-run agent); fs-extra was neither an outright
# nor a validation failure, so the numerators are unchanged.
EXPECTED = {
    "TestPilot-500": {
        "denominator": 20,
        "outright": 1,
        "later": 4,
        "total": 5,
    },
    "TestPilot-1000": {
        "denominator": 20,
        "outright": 0,
        "later": 6,
        "total": 6,
    },
    "Agent full+residual T1": {
        "denominator": 40,
        "outright": 3,
        "later": 1,
        "total": 4,
    },
    "Agent full+residual T0": {
        "denominator": 20,
        "outright": 0,
        "later": 0,
        "total": 0,
    },
}


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def pct(numer, denom):
    return 100.0 * numer / denom if denom else 0.0


def fmt_cell(numer, denom):
    return f"{numer}/{denom} ({pct(numer, denom):.1f}\\%)"


def report_tests(report):
    return list(report.get("tests") or [])


def collect_testpilot_row(label, run_dir):
    denominator = len(LIBS)
    outright_libs = []
    later_libs = []

    for lib in comparison.DOMAIN_MAP:
        report_path = os.path.join(run_dir, lib, "report.json")
        if not os.path.exists(report_path):
            sys.exit(f"missing TestPilot report: {report_path}")

        tests = report_tests(read_json(report_path))
        passing = sum(1 for test in tests if test.get("status") == "PASSED")
        discarded = sum(
            1
            for test in tests
            if test.get("status") == "PASSED"
            and test.get("dryRunStatus") == "failed"
        )
        retained = passing - discarded

        if retained == 0:
            outright_libs.append(lib)
        if discarded > 0 and retained > 0:
            later_libs.append(lib)

    total_libs = sorted(set(outright_libs) | set(later_libs))
    return {
        "system": label,
        "denominator": denominator,
        "outright": len(outright_libs),
        "later": len(later_libs),
        "total": len(total_libs),
        "outright_items": ";".join(outright_libs),
        "later_items": ";".join(later_libs),
        "total_items": ";".join(total_libs),
    }


def agent_category(run):
    if run["condition"] == "hidden":
        return "Agent hidden T1"
    if run["condition"] == "full+residual" and run["temperature"] == "1":
        return "Agent full+residual T1"
    if run["condition"] == "full+residual" and run["temperature"] == "0":
        return "Agent full+residual T0"
    sys.exit(
        "unknown agent run category: "
        f'{run["run_id"]} condition={run["condition"]} temperature={run["temperature"]}'
    )


def collect_agent_rows():
    categories = {
        "Agent full+residual T1": [],
        "Agent full+residual T0": [],
    }
    for run in agent_nontriviality.configured_runs():
        category = agent_category(run)
        if category in categories:
            categories[category].append(run)

    rows = []
    for label, runs in categories.items():
        rows.append(collect_agent_category_row(label, runs))
    return rows


def collect_agent_category_row(label, runs):
    denominator = len(runs) * len(LIBS)
    outright = []
    later = []

    for run in runs:
        for lib in LIBS:
            report_path = os.path.join(run["run_dir"], lib, "report", "report.json")
            if not os.path.exists(report_path):
                sys.exit(f"missing agent report: {report_path}")

            tests = report_tests(read_json(report_path))
            passing = sum(1 for test in tests if test.get("status") == "PASSED")
            failing = sum(1 for test in tests if test.get("status") == "FAILED")
            key = f'{run["run_id"]}:{lib}'

            if len(tests) == 0 or passing == 0:
                outright.append(key)
            elif failing > 0:
                later.append(key)

    total_items = sorted(set(outright) | set(later))
    return {
        "system": label,
        "denominator": denominator,
        "outright": len(outright),
        "later": len(later),
        "total": len(total_items),
        "outright_items": ";".join(outright),
        "later_items": ";".join(later),
        "total_items": ";".join(total_items),
    }


def validate(rows):
    # Golden regression guard: expectations below match the shipped data
    # snapshot and sys.exit on mismatch. Update them deliberately when the
    # data is regenerated.
    errors = []
    for row in rows:
        expected = EXPECTED.get(row["system"])
        if expected is None:
            errors.append(f'unexpected row: {row["system"]}')
            continue
        for key, want in expected.items():
            got = row[key]
            if got != want:
                errors.append(f'{row["system"]} {key}: expected {want}, got {got}')
    if errors:
        sys.exit("correctness failure table validation failed:\n" + "\n".join(errors))


def write_csv(rows):
    os.makedirs(TABLES_DIR, exist_ok=True)
    fields = [
        "system",
        "denominator",
        "outright",
        "later",
        "total",
        "outright_items",
        "later_items",
        "total_items",
    ]
    with open(OUTPUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_tex(rows):
    lines = [
        WARN,
        r"\begin{table}[h]",
        r"  \centering",
        r"  \caption{Number of runs affected by validation "
        r"failures. ``Validation failures'' are attempts that produced tests "
        r"but failed later validation. ``T0'' and ``T1'' refer to the agent "
        r"experiments at temperature 0.0 and 1.0, respectively.}",
        r"  \label{tab:gpt5-correctness-failure-overview}",
        r"  \small",
        r"  \begin{tabular}{lrrr}",
        r"  \toprule",
        r"  \textbf{System} &",
        r"  \textbf{No passing tests} &",
        r"  \textbf{Validation failures} &",
        r"  \textbf{Total affected} \\",
        r"  \midrule",
    ]
    for row in rows:
        denom = row["denominator"]
        lines.append(
            f'{row["system"]} & '
            f'{fmt_cell(row["outright"], denom)} & '
            f'{fmt_cell(row["later"], denom)} & '
            f'{fmt_cell(row["total"], denom)} \\\\'
        )
    lines.extend([r"  \bottomrule", r"  \end{tabular}", r"\end{table}", ""])

    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(OUTPUT_TEX, "w") as fh:
        fh.write("\n".join(lines))


def main():
    rows = [collect_testpilot_row(label, run_dir) for label, run_dir in TESTPILOT_RUNS]
    rows.extend(collect_agent_rows())
    validate(rows)
    write_csv(rows)
    write_tex(rows)

    print(f"wrote {OUTPUT_TEX}")
    print(f"wrote {OUTPUT_CSV}")
    for row in rows:
        denom = row["denominator"]
        print(
            f'{row["system"]}: outright {fmt_cell(row["outright"], denom)}, '
            f'later {fmt_cell(row["later"], denom)}, '
            f'total {fmt_cell(row["total"], denom)}'
        )


if __name__ == "__main__":
    main()
