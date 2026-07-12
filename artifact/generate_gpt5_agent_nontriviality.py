#!/usr/bin/env python3
"""
Generate non-triviality data for the five GPT-5.4 agent runs.

The original TestPilot non-triviality query expects generated tests to import
the package under test by package name. The agent archives generated tests that
usually import the checked-out package via relative paths, so the CodeQL query
emits both:
  - strict: TestPilot-style package-name import dependency
  - agent_compatible: strict OR relative-import dependency

Outputs:
  tables/tab-gpt5-agent-nontriviality.csv
  tables/tab-gpt5-agent-nontriviality-tests.csv
  tables/tab-gpt5-agent-nontriviality-summary.tex
"""

import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True

import generate_gpt5_agent_residual_table as residual


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR = os.path.join(SCRIPT_DIR, "tables")
QUERY_FILE = os.path.join(SCRIPT_DIR, "ql", "queries", "AgentNonTriviality.ql")
WARN = "%WARNING: auto-generated. Edit generate_gpt5_agent_nontriviality.py instead.\n"

TEMP0_RUN_DEFAULT = os.path.join(
    SCRIPT_DIR,
    "data",
    "gpt-5.4-agent-residual",
    "head-63be77f",
    "residual-full-gpt54-temp0-20260530",
)

RUNS = [
    {
        "run_id": "R3",
        "run_label": "Full T1 R3",
        "condition": "full+residual",
        "temperature": "1",
        "run_dir": os.path.join(
            SCRIPT_DIR,
            "data",
            "gpt-5.4-agent-residual",
            residual.RUN_DIRS["3"],
        ),
    },
    {
        "run_id": "R4",
        "run_label": "Full T1 R4",
        "condition": "full+residual",
        "temperature": "1",
        "run_dir": os.path.join(
            SCRIPT_DIR,
            "data",
            "gpt-5.4-agent-residual",
            residual.RUN_DIRS["4"],
        ),
    },
    {
        "run_id": "H1",
        "run_label": "Hidden T1 H1",
        "condition": "hidden",
        "temperature": "1",
        "run_dir": os.path.join(SCRIPT_DIR, "data", "gpt-5.4-agent-hidden", "1"),
    },
    {
        "run_id": "H2",
        "run_label": "Hidden T1 H2",
        "condition": "hidden",
        "temperature": "1",
        "run_dir": os.path.join(
            SCRIPT_DIR,
            "data",
            "gpt-5.4-agent-hidden",
            "spliced-20260523-fs-extra-redo",
        ),
    },
    {
        "run_id": "T0",
        "run_label": "Full T0",
        "condition": "full+residual",
        "temperature": "0",
        "run_dir": None,
    },
]


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def pct(numer, denom):
    if denom == 0:
        return None
    return 100.0 * numer / denom


def fmt_pct(value):
    if value is None:
        return "--"
    return f"{value:.1f}\\%"


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


def configured_runs():
    runs = [dict(run) for run in RUNS]
    for run in runs:
        if run["run_id"] == "T0":
            run["run_dir"] = resolve_temp0_run_dir()
        run["run_dir"] = os.path.abspath(run["run_dir"])
    return runs


def suite_statuses(run_dir):
    path = os.path.join(run_dir, "suite-report.json")
    if not os.path.exists(path):
        return {}
    suite = read_json(path)
    return {
        entry.get("libraryId"): entry.get("status")
        for entry in suite.get("libraries", [])
    }


def test_file_name(test_entry):
    return test_entry.get("testFile") or test_entry.get("testName")


def collect_inputs(runs):
    library_rows = {}
    test_rows = {}
    missing = []

    for run in runs:
        statuses = suite_statuses(run["run_dir"])
        for lib in residual.DOMAIN_MAP:
            report_path = os.path.join(run["run_dir"], lib, "report", "report.json")
            tests_dir = os.path.join(run["run_dir"], lib, "report", "tests")
            if not os.path.exists(report_path):
                missing.append(report_path)
                continue
            report = read_json(report_path)
            tests = list(report.get("tests") or [])
            row_key = (run["run_id"], lib)
            library_rows[row_key] = {
                **run,
                "library": lib,
                "package_name": report.get("packageName") or lib,
                "status": statuses.get(lib),
                # repo-relative so the CSV ships without machine-specific paths
                "report_path": os.path.relpath(report_path, SCRIPT_DIR),
                "tests_dir": tests_dir,
                "tests_total": len(tests),
                "tests_passing": sum(1 for t in tests if t.get("status") == "PASSED"),
                "tests_failing": sum(1 for t in tests if t.get("status") == "FAILED"),
                "tests_other": sum(
                    1 for t in tests if t.get("status") not in {"PASSED", "FAILED"}
                ),
            }
            for test in tests:
                name = test_file_name(test)
                key = (run["run_id"], lib, name)
                test_rows[key] = {
                    **run,
                    "library": lib,
                    "package_name": report.get("packageName") or lib,
                    "status_library": statuses.get(lib),
                    "test_file": name,
                    "test_name": test.get("testName"),
                    "test_status": test.get("status"),
                    "strict_nontrivial": False,
                    "agent_compatible_nontrivial": False,
                    "strict_import_paths": set(),
                    "agent_compatible_import_paths": set(),
                }

    if missing:
        sys.exit(
            "missing report.json files; refusing to generate incomplete non-triviality data:\n"
            + "\n".join(f"- {path}" for path in missing)
        )
    return library_rows, test_rows


def write_minimized_source_tree(runs, source_root):
    for run in runs:
        for lib in residual.DOMAIN_MAP:
            report_path = os.path.join(run["run_dir"], lib, "report", "report.json")
            tests_dir = os.path.join(run["run_dir"], lib, "report", "tests")
            report = read_json(report_path)
            package_name = report.get("packageName") or lib

            dest_dir = os.path.join(source_root, run["run_id"], lib)
            dest_tests_dir = os.path.join(dest_dir, "tests")
            os.makedirs(dest_tests_dir, exist_ok=True)

            minimized = {
                "packageName": package_name,
                "metaData": {"packageName": package_name},
                "stats": report.get("stats") or {},
                "tests": [
                    {
                        "testName": test.get("testName"),
                        "testFile": test_file_name(test),
                        "status": test.get("status"),
                        "coveredStatements": test.get("coveredStatements") or [],
                        "err": test.get("err"),
                    }
                    for test in report.get("tests") or []
                ],
                "coverage": {"total": ((report.get("coverage") or {}).get("total") or {})},
            }
            with open(os.path.join(dest_dir, "report.json"), "w") as fh:
                json.dump(minimized, fh, sort_keys=True)

            if os.path.isdir(tests_dir):
                for name in os.listdir(tests_dir):
                    src = os.path.join(tests_dir, name)
                    if os.path.isfile(src) and name.endswith((".js", ".cjs", ".mjs")):
                        shutil.copy2(src, os.path.join(dest_tests_dir, name))


def run_command(cmd, *, env=None, cwd=None):
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        tail = result.stdout[-6000:] if result.stdout else ""
        sys.exit(
            "command failed:\n"
            + " ".join(cmd)
            + ("\n\noutput tail:\n" + tail if tail else "")
        )
    return result.stdout


def codeql_search_path():
    candidates = ["/usr/local/codeql-home/codeql-repo", os.path.join(SCRIPT_DIR, "ql")]
    return ":".join(path for path in candidates if os.path.exists(path))


def run_codeql(source_root, work_dir):
    codeql = shutil.which("codeql")
    if not codeql:
        sys.exit("codeql command not found; install CodeQL or add it to PATH")

    db_dir = os.path.join(work_dir, "db")
    bqrs_path = os.path.join(work_dir, "AgentNonTriviality.bqrs")
    decoded_path = os.path.join(work_dir, "AgentNonTriviality.csv")

    env = os.environ.copy()
    env["LGTM_INDEX_FILTERS"] = "include:**/*.json\nexclude:**/coverageData/**/*.json"
    env["LGTM_MAX_FILE_SIZE"] = "20"

    run_command([
        codeql,
        "database",
        "create",
        "--overwrite",
        "--threads",
        "3",
        "-l",
        "javascript",
        "--source-root",
        source_root,
        "--",
        db_dir,
    ], env=env)

    run_command([
        codeql,
        "query",
        "run",
        "--threads",
        "3",
        "--search-path",
        codeql_search_path(),
        "--output",
        bqrs_path,
        "-d",
        db_dir,
        QUERY_FILE,
    ])

    run_command([
        codeql,
        "bqrs",
        "decode",
        "-r",
        "agentNonTriviality",
        "--format",
        "csv",
        "--output",
        decoded_path,
        bqrs_path,
    ])
    return decoded_path


def normalize_codeql_path(path):
    return path.replace("\\", "/").lstrip("/")


def load_codeql_rows(decoded_path):
    with open(decoded_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows


def apply_codeql_rows(rows, test_rows):
    unmatched = []
    for row in rows:
        test_path = normalize_codeql_path(row["testPath"])
        parts = test_path.split("/")
        if len(parts) < 4 or parts[2] != "tests":
            unmatched.append((test_path, "unexpected path shape"))
            continue
        run_id, lib, _, test_file = parts[0], parts[1], parts[2], "/".join(parts[3:])
        key = (run_id, lib, test_file)
        if key not in test_rows:
            unmatched.append((test_path, "not present in report tests array"))
            continue
        definition = row["definition"]
        import_path = row["importPath"]
        if definition == "strict":
            test_rows[key]["strict_nontrivial"] = True
            test_rows[key]["strict_import_paths"].add(import_path)
        elif definition == "agent_compatible":
            test_rows[key]["agent_compatible_nontrivial"] = True
            test_rows[key]["agent_compatible_import_paths"].add(import_path)
        else:
            unmatched.append((test_path, f"unexpected definition {definition!r}"))

    if unmatched:
        shown = "\n".join(f"- {path}: {reason}" for path, reason in unmatched[:20])
        more = f"\n... {len(unmatched) - 20} more" if len(unmatched) > 20 else ""
        sys.exit("CodeQL rows did not map cleanly to report tests:\n" + shown + more)


def aggregate_library_rows(library_rows, test_rows):
    for row in library_rows.values():
        tests = [
            t
            for key, t in test_rows.items()
            if key[0] == row["run_id"] and key[1] == row["library"]
        ]
        strict_tests = [t for t in tests if t["strict_nontrivial"]]
        agent_tests = [t for t in tests if t["agent_compatible_nontrivial"]]
        strict_passes = [
            t for t in strict_tests if t["test_status"] == "PASSED"
        ]
        agent_passes = [
            t for t in agent_tests if t["test_status"] == "PASSED"
        ]
        row["nontrivial_tests_strict"] = len(strict_tests)
        row["nontrivial_passes_strict"] = len(strict_passes)
        row["nontrivial_test_rate_strict"] = pct(len(strict_tests), row["tests_total"])
        row["nontrivial_pass_rate_strict"] = pct(
            len(strict_passes), row["tests_passing"]
        )
        row["nontrivial_tests_agent"] = len(agent_tests)
        row["nontrivial_passes_agent"] = len(agent_passes)
        row["nontrivial_test_rate_agent"] = pct(len(agent_tests), row["tests_total"])
        row["nontrivial_pass_rate_agent"] = pct(
            len(agent_passes), row["tests_passing"]
        )


def summarize_runs(library_rows):
    summaries = []
    run_order = [run["run_id"] for run in configured_runs()]
    for run_id in run_order:
        rows = [row for row in library_rows.values() if row["run_id"] == run_id]
        if not rows:
            continue
        first = rows[0]
        summary = {
            "run_id": run_id,
            "run_label": first["run_label"],
            "condition": first["condition"],
            "temperature": first["temperature"],
            "libraries": len(rows),
            "tests_total": sum(row["tests_total"] for row in rows),
            "tests_passing": sum(row["tests_passing"] for row in rows),
            "tests_failing": sum(row["tests_failing"] for row in rows),
            "nontrivial_tests_strict": sum(
                row["nontrivial_tests_strict"] for row in rows
            ),
            "nontrivial_passes_strict": sum(
                row["nontrivial_passes_strict"] for row in rows
            ),
            "nontrivial_tests_agent": sum(
                row["nontrivial_tests_agent"] for row in rows
            ),
            "nontrivial_passes_agent": sum(
                row["nontrivial_passes_agent"] for row in rows
            ),
        }
        summary["nontrivial_pass_rate_strict"] = pct(
            summary["nontrivial_passes_strict"], summary["tests_passing"]
        )
        summary["nontrivial_pass_rate_agent"] = pct(
            summary["nontrivial_passes_agent"], summary["tests_passing"]
        )
        summaries.append(summary)
    return summaries


def write_library_csv(library_rows, path):
    fields = [
        "run_id",
        "run_label",
        "condition",
        "temperature",
        "library",
        "package_name",
        "status",
        "tests_total",
        "tests_passing",
        "tests_failing",
        "tests_other",
        "nontrivial_tests_strict",
        "nontrivial_passes_strict",
        "nontrivial_test_rate_strict",
        "nontrivial_pass_rate_strict",
        "nontrivial_tests_agent",
        "nontrivial_passes_agent",
        "nontrivial_test_rate_agent",
        "nontrivial_pass_rate_agent",
        "report_path",
    ]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for key in sorted(library_rows):
            writer.writerow({field: library_rows[key].get(field) for field in fields})


def write_tests_csv(test_rows, path):
    fields = [
        "run_id",
        "run_label",
        "condition",
        "temperature",
        "library",
        "package_name",
        "status_library",
        "test_file",
        "test_name",
        "test_status",
        "strict_nontrivial",
        "agent_compatible_nontrivial",
        "strict_import_paths",
        "agent_compatible_import_paths",
    ]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for key in sorted(test_rows):
            row = dict(test_rows[key])
            row["strict_import_paths"] = ";".join(sorted(row["strict_import_paths"]))
            row["agent_compatible_import_paths"] = ";".join(
                sorted(row["agent_compatible_import_paths"])
            )
            writer.writerow({field: row.get(field) for field in fields})


def write_summary_table(summaries, path):
    lines = [
        WARN.rstrip("\n"),
        r"\begin{table}[t!]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Non-triviality of generated tests across the five GPT-5.4 "
        r"agent runs. Strict uses the original TestPilot package-name import "
        r"definition; Agent-compatible also counts assertions depending on "
        r"relative imports of the archived package under test. The full "
        r"per-library and per-test audit data are in "
        r"\texttt{tab-gpt5-agent-nontriviality.csv} and "
        r"\texttt{tab-gpt5-agent-nontriviality-tests.csv}.}",
        r"\label{tab:gpt5-agent-nontriviality-summary}",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"\textbf{Run} & \textbf{Condition} & \textbf{Passing} & "
        r"\textbf{Strict NT} & \textbf{Agent NT} & \textbf{Agent NT rate} \\",
        r"\midrule",
    ]
    for row in summaries:
        condition = (
            row["condition"].replace("+", r"$+$") + f", T={row['temperature']}"
        )
        lines.append(
            f"{row['run_id']} & {condition} & "
            f"{row['tests_passing']} & "
            f"{row['nontrivial_passes_strict']} & "
            f"{row['nontrivial_passes_agent']} & "
            f"{fmt_pct(row['nontrivial_pass_rate_agent'])} \\\\"
        )
    strict_rates = [r["nontrivial_pass_rate_strict"] for r in summaries]
    agent_rates = [r["nontrivial_pass_rate_agent"] for r in summaries]
    lines += [
        r"\midrule",
        r"\textbf{Median} & & "
        f"{statistics.median([r['tests_passing'] for r in summaries]):.0f} & "
        f"{statistics.median([r['nontrivial_passes_strict'] for r in summaries]):.0f} & "
        f"{statistics.median([r['nontrivial_passes_agent'] for r in summaries]):.0f} & "
        f"{fmt_pct(statistics.median(agent_rates))} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def validate(library_rows, test_rows, summaries):
    # Golden regression guard: expectations below match the shipped data
    # snapshot and sys.exit on mismatch. Update them deliberately when the
    # data is regenerated.
    expected_reports = len(RUNS) * len(residual.DOMAIN_MAP)
    if len(library_rows) != expected_reports:
        sys.exit(f"expected {expected_reports} library reports, got {len(library_rows)}")
    if len(summaries) != len(RUNS):
        sys.exit(f"expected {len(RUNS)} run summaries, got {len(summaries)}")
    if not test_rows:
        sys.exit("no generated tests found in archived reports")
    if sum(row["nontrivial_passes_agent"] for row in summaries) == 0:
        sys.exit("CodeQL produced no agent-compatible non-trivial passing tests")
    for row in library_rows.values():
        if row["nontrivial_tests_agent"] < row["nontrivial_tests_strict"]:
            sys.exit(f"agent-compatible count below strict count for {row['run_id']} {row['library']}")
        if row["nontrivial_passes_agent"] > row["tests_passing"]:
            sys.exit(f"agent-compatible passes exceed passing tests for {row['run_id']} {row['library']}")
    for row in test_rows.values():
        if row["strict_nontrivial"] and not row["agent_compatible_nontrivial"]:
            sys.exit(
                f"strict non-trivial test is not agent-compatible: "
                f"{row['run_id']} {row['library']} {row['test_file']}"
            )


def print_summary(summaries, library_csv, tests_csv, table_path, tmp_dir=None):
    print(f"wrote {os.path.relpath(library_csv, SCRIPT_DIR)}")
    print(f"wrote {os.path.relpath(tests_csv, SCRIPT_DIR)}")
    print(f"wrote {os.path.relpath(table_path, SCRIPT_DIR)}")
    if tmp_dir:
        print(f"kept temporary CodeQL workspace: {tmp_dir}")
    print()
    print("run totals:")
    for row in summaries:
        print(
            f"  {row['run_id']}: passing={row['tests_passing']}, "
            f"strict_nt={row['nontrivial_passes_strict']} "
            f"({row['nontrivial_pass_rate_strict']:.1f}%), "
            f"agent_nt={row['nontrivial_passes_agent']} "
            f"({row['nontrivial_pass_rate_agent']:.1f}%)"
        )
    print(
        "  median agent-compatible pass rate: "
        f"{statistics.median([r['nontrivial_pass_rate_agent'] for r in summaries]):.1f}%"
    )


def generate(work_dir):
    runs = configured_runs()
    library_rows, test_rows = collect_inputs(runs)

    source_root = os.path.join(work_dir, "source")
    os.makedirs(source_root, exist_ok=True)
    write_minimized_source_tree(runs, source_root)
    decoded_path = run_codeql(source_root, work_dir)
    codeql_rows = load_codeql_rows(decoded_path)
    apply_codeql_rows(codeql_rows, test_rows)
    aggregate_library_rows(library_rows, test_rows)
    summaries = summarize_runs(library_rows)
    validate(library_rows, test_rows, summaries)

    os.makedirs(TABLES_DIR, exist_ok=True)
    library_csv = os.path.join(TABLES_DIR, "tab-gpt5-agent-nontriviality.csv")
    tests_csv = os.path.join(TABLES_DIR, "tab-gpt5-agent-nontriviality-tests.csv")
    table_path = os.path.join(TABLES_DIR, "tab-gpt5-agent-nontriviality-summary.tex")
    write_library_csv(library_rows, library_csv)
    write_tests_csv(test_rows, tests_csv)
    write_summary_table(summaries, table_path)
    return summaries, library_csv, tests_csv, table_path


def main():
    if not os.path.exists(QUERY_FILE):
        sys.exit(f"missing CodeQL query: {QUERY_FILE}")
    if os.environ.get("KEEP_NONTRIVIAL_TMP"):
        work_dir = tempfile.mkdtemp(prefix="gpt5-agent-nontriviality-")
        summaries, library_csv, tests_csv, table_path = generate(work_dir)
        print_summary(summaries, library_csv, tests_csv, table_path, work_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="gpt5-agent-nontriviality-") as work_dir:
            summaries, library_csv, tests_csv, table_path = generate(work_dir)
            print_summary(summaries, library_csv, tests_csv, table_path)


if __name__ == "__main__":
    main()
