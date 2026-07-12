#!/usr/bin/env python3
"""
Generate the GPT-5.4 hidden mutation-feedback ablation table.

Reads:
  data/gpt-5.4-agent-hidden/1/
  data/gpt-5.4-agent-hidden/spliced-20260523-fs-extra-redo/
  data/gpt-5.4-agent-residual/head-63be77f/{R3,R4}/ via generate_gpt5_agent_residual_table.py

The hidden condition is compared against the full condition's pre-residual
aggregate score. This isolates the effect of model-visible per-item mutation
feedback without crediting the residual phase. The comparison is unpaired:
hidden and full are independent agent runs, so the feedback delta is computed
as the difference between condition means.
"""

import csv
import json
import os
import statistics

import generate_gpt5_agent_residual_table as residual

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HIDDEN_RUN_IDS = ["1", "spliced-20260523-fs-extra-redo"]
HIDDEN_RUN_LABELS = {"1": "H1", "spliced-20260523-fs-extra-redo": "H2"}
HIDDEN_DATA_DIR = os.path.join(SCRIPT_DIR, "data", "gpt-5.4-agent-hidden")
TABLES_DIR = os.path.join(SCRIPT_DIR, "tables")

WARN = "%WARNING: auto-generated. Edit generate_gpt5_agent_hidden_ablation_table.py instead.\n"
STAR = r"\textsuperscript{*}"


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def fmt_tests(passing, written):
    if passing is None and written is None:
        return "--"
    return f"{passing if passing is not None else '--'}/{written if written is not None else '--'}"


def fmt_status(status, complete):
    if complete:
        return "ok"
    if status in {"failed_setup", "failed_benchmark"}:
        return "fail"
    if status == "skipped":
        return "--"
    if status in {"completed", "partial_benchmark"}:
        return "fail"
    return status or "--"


def fmt_status_pair(value):
    return "/".join(value)


def fmt_pct(value, star=False):
    if value is None:
        return "--"
    return f"{value:.1f}\\%" + (STAR if star else "")


def fmt_delta(value, star=False):
    if value is None:
        return "--"
    sign = "+" if value >= 0 else "-"
    return f"${sign}${abs(value):.1f}" + (STAR if star else "")


def should_star(*counts):
    return any(0 < count < len(HIDDEN_RUN_IDS) for count in counts)


def hidden_cell(run_id, lib):
    run_dir = os.path.join(HIDDEN_DATA_DIR, run_id)
    suite = read_json(os.path.join(run_dir, "suite-report.json")) or {}
    suite_status = {
        e.get("libraryId"): e.get("status")
        for e in suite.get("libraries", [])
    }
    report = read_json(os.path.join(run_dir, lib, "report", "report.json"))
    if report is None:
        return {
            "status": suite_status.get(lib, "--"),
            "complete": False,
            "tests_written": None,
            "tests_passing": None,
            "hidden_score": None,
            "hidden_total_valid": None,
            "hidden_undetected": None,
            "hidden_benchmark_score": None,
            "hidden_benchmark_total_valid": None,
            "hidden_benchmark_undetected": None,
        }

    ar = report.get("agentRun", {})
    gen = ar.get("generation", {})
    agg = gen.get("aggregatePass") or {}
    stats = report.get("stats") or {}
    stryker = report.get("stryker") or {}
    metrics = stryker.get("metrics") or {}

    complete = (
        suite_status.get(lib) == "completed"
        and not ar.get("fatalError")
        and (stats.get("nrFailures") or 0) == 0
        and agg.get("succeeded")
        and stryker.get("exitCode") == 0
        and bool(metrics)
    )

    cell = {
        "status": suite_status.get(lib),
        "complete": complete,
        "tests_written": gen.get("testsWritten"),
        "tests_passing": stats.get("nrPasses"),
        "hidden_score": agg.get("mutationScore") if agg.get("succeeded") else None,
        "hidden_total_valid": agg.get("totalValid") if agg.get("succeeded") else None,
        "hidden_undetected": agg.get("undetectedCount") if agg.get("succeeded") else None,
        "hidden_benchmark_score": metrics.get("mutationScore") if metrics else None,
        "hidden_benchmark_total_valid": metrics.get("totalValid") if metrics else None,
        "hidden_benchmark_undetected": metrics.get("totalUndetected") if metrics else None,
    }
    if complete:
        return cell
    for key in [
        "hidden_score",
        "hidden_total_valid",
        "hidden_undetected",
        "hidden_benchmark_score",
        "hidden_benchmark_total_valid",
        "hidden_benchmark_undetected",
    ]:
        cell[key] = None
    return cell


def collect_rows():
    rates = residual.load_pricing()
    full_runs = {r: residual.collect_run(r) for r in residual.RUN_IDS}
    full_rows = residual.aggregate(full_runs, rates)
    hidden_runs = {
        run_id: {lib: hidden_cell(run_id, lib) for lib in residual.DOMAIN_MAP}
        for run_id in HIDDEN_RUN_IDS
    }

    rows = {}
    for lib in residual.DOMAIN_MAP:
        hidden_cells = [hidden_runs[run_id][lib] for run_id in HIDDEN_RUN_IDS]
        hidden_score, hidden_score_n = residual.avg([
            c.get("hidden_score") for c in hidden_cells
        ])
        hidden_benchmark_score, hidden_benchmark_score_n = residual.avg([
            c.get("hidden_benchmark_score") for c in hidden_cells
        ])
        hidden_total_valid, hidden_total_valid_n = residual.avg([
            c.get("hidden_total_valid") for c in hidden_cells
        ])
        hidden_undetected, hidden_undetected_n = residual.avg([
            c.get("hidden_undetected") for c in hidden_cells
        ])
        hidden_benchmark_total_valid, hidden_benchmark_total_valid_n = residual.avg([
            c.get("hidden_benchmark_total_valid") for c in hidden_cells
        ])
        hidden_benchmark_undetected, hidden_benchmark_undetected_n = residual.avg([
            c.get("hidden_benchmark_undetected") for c in hidden_cells
        ])

        full_pre = full_rows[lib]["pre_score"]
        full_n = full_rows[lib]["pre_score_n"]
        delta = (
            full_pre - hidden_score
            if full_pre is not None and hidden_score is not None
            else None
        )
        rows[lib] = {
            "hidden_cells": hidden_cells,
            "status_pair": [fmt_status(c.get("status"), c.get("complete"))
                            for c in hidden_cells],
            "full_status_pair": [
                residual.display_status(full_runs[run_id].get(lib))
                for run_id in residual.RUN_IDS
            ],
            "hidden_score": hidden_score,
            "hidden_score_n": hidden_score_n,
            "hidden_total_valid": hidden_total_valid,
            "hidden_total_valid_n": hidden_total_valid_n,
            "hidden_undetected": hidden_undetected,
            "hidden_undetected_n": hidden_undetected_n,
            "hidden_benchmark_score": hidden_benchmark_score,
            "hidden_benchmark_score_n": hidden_benchmark_score_n,
            "hidden_benchmark_total_valid": hidden_benchmark_total_valid,
            "hidden_benchmark_total_valid_n": hidden_benchmark_total_valid_n,
            "hidden_benchmark_undetected": hidden_benchmark_undetected,
            "hidden_benchmark_undetected_n": hidden_benchmark_undetected_n,
            "full_pre_score": full_pre,
            "full_pre_n": full_n,
            "delta_feedback": delta,
            "delta_feedback_star": should_star(hidden_score_n, full_n),
        }
    return rows, hidden_runs


def write_scores_table(rows, path):
    lines = [
        WARN.rstrip("\n"),
        r"\begin{table*}[t!]",
        r"\centering",
        r"\caption{Hidden mutation-feedback ablation for GPT-5.4. Hidden is the "
        r"\texttt{hidden}-condition final score; Full pre-residual is the "
        r"full-feedback score before residual feedback contributes. "
        r"$\Delta_{\mathrm{feedback}}$ is Full pre-residual $-$ Hidden "
        r"(unpaired difference of means). Full-feedback no-test failures "
        r"follow the residual table and contribute 0.0\%. "
        r"Cells marked \textsuperscript{*} use only one run.}",
        r"\label{tab:gpt5-agent-hidden-ablation-scores}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l r r r}",
        r"\toprule",
        r"\textbf{Project} & "
        r"\textbf{Hidden mutation score} & "
        r"\textbf{Full pre-residual mutation score} & "
        r"\textbf{$\Delta_{\mathrm{feedback}}$ (pp)} \\",
        r"\midrule",
    ]
    for lib in residual.DOMAIN_MAP:
        if lib in residual.EXCLUDE_LIBS:
            continue
        row = rows[lib]
        hidden_star = should_star(row["hidden_score_n"])
        full_star = 0 < row["full_pre_n"] < len(residual.RUN_IDS)
        lines.append(
            f"{lib} & "
            + " & ".join([
                fmt_pct(row.get("hidden_score"), hidden_star),
                fmt_pct(row.get("full_pre_score"), full_star),
                fmt_delta(row.get("delta_feedback"),
                          row.get("delta_feedback_star")),
            ])
            + r" \\"
        )
        if lib in residual.SEPARATOR_LIBS:
            lines.append(r"\midrule")

    hidden_scores = [rows[lib]["hidden_score"] for lib in residual.DOMAIN_MAP
                     if lib not in residual.EXCLUDE_LIBS
                     and rows[lib]["hidden_score"] is not None]
    full_scores = [rows[lib]["full_pre_score"] for lib in residual.DOMAIN_MAP
                   if lib not in residual.EXCLUDE_LIBS
                   and rows[lib]["full_pre_score"] is not None]
    deltas = [rows[lib]["delta_feedback"] for lib in residual.DOMAIN_MAP
              if lib not in residual.EXCLUDE_LIBS
              and rows[lib]["delta_feedback"] is not None]

    def _agg(values, fn):
        return fn(values) if values else None

    lines.append(r"\midrule")
    lines.append(
        r"\textbf{Mean} & "
        + fmt_pct(_agg(hidden_scores, statistics.mean)) + " & "
        + fmt_pct(_agg(full_scores, statistics.mean)) + " & "
        + fmt_delta(_agg(deltas, statistics.mean))
        + r" \\"
    )
    lines.append(
        r"\textbf{Median} & "
        + fmt_pct(_agg(hidden_scores, statistics.median)) + " & "
        + fmt_pct(_agg(full_scores, statistics.median)) + " & "
        + fmt_delta(_agg(deltas, statistics.median))
        + r" \\"
    )
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def write_csv(rows, path):
    fields = [
        "library",
        "status_h1",
        "status_h2",
        "display_status_h1",
        "display_status_h2",
        "complete_h1",
        "complete_h2",
        "tests_passing_h1",
        "tests_passing_h2",
        "tests_written_h1",
        "tests_written_h2",
        "hidden_score_h1",
        "hidden_score_h2",
        "hidden_score",
        "hidden_score_n",
        "hidden_total_valid_h1",
        "hidden_total_valid_h2",
        "hidden_total_valid",
        "hidden_total_valid_n",
        "hidden_undetected_h1",
        "hidden_undetected_h2",
        "hidden_undetected",
        "hidden_undetected_n",
        "hidden_benchmark_score_h1",
        "hidden_benchmark_score_h2",
        "hidden_benchmark_score",
        "hidden_benchmark_score_n",
        "hidden_benchmark_total_valid_h1",
        "hidden_benchmark_total_valid_h2",
        "hidden_benchmark_total_valid",
        "hidden_benchmark_total_valid_n",
        "hidden_benchmark_undetected_h1",
        "hidden_benchmark_undetected_h2",
        "hidden_benchmark_undetected",
        "hidden_benchmark_undetected_n",
        "display_status_full_r1",
        "display_status_full_r2",
        "full_pre_score",
        "full_pre_n",
        "delta_feedback",
    ]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for lib in residual.DOMAIN_MAP:
            row = rows[lib]
            h1, h2 = row["hidden_cells"]
            out = {
                "library": lib,
                "status_h1": h1.get("status"),
                "status_h2": h2.get("status"),
                "display_status_h1": row["status_pair"][0],
                "display_status_h2": row["status_pair"][1],
                "display_status_full_r1": row["full_status_pair"][0],
                "display_status_full_r2": row["full_status_pair"][1],
                "complete_h1": h1.get("complete"),
                "complete_h2": h2.get("complete"),
                "tests_passing_h1": h1.get("tests_passing"),
                "tests_passing_h2": h2.get("tests_passing"),
                "tests_written_h1": h1.get("tests_written"),
                "tests_written_h2": h2.get("tests_written"),
                "hidden_score_h1": h1.get("hidden_score"),
                "hidden_score_h2": h2.get("hidden_score"),
                "hidden_total_valid_h1": h1.get("hidden_total_valid"),
                "hidden_total_valid_h2": h2.get("hidden_total_valid"),
                "hidden_undetected_h1": h1.get("hidden_undetected"),
                "hidden_undetected_h2": h2.get("hidden_undetected"),
                "hidden_benchmark_score_h1": h1.get("hidden_benchmark_score"),
                "hidden_benchmark_score_h2": h2.get("hidden_benchmark_score"),
                "hidden_benchmark_total_valid_h1": h1.get("hidden_benchmark_total_valid"),
                "hidden_benchmark_total_valid_h2": h2.get("hidden_benchmark_total_valid"),
                "hidden_benchmark_undetected_h1": h1.get("hidden_benchmark_undetected"),
                "hidden_benchmark_undetected_h2": h2.get("hidden_benchmark_undetected"),
                **{k: row.get(k) for k in [
                    "hidden_score",
                    "hidden_score_n",
                    "hidden_total_valid",
                    "hidden_total_valid_n",
                    "hidden_undetected",
                    "hidden_undetected_n",
                    "hidden_benchmark_score",
                    "hidden_benchmark_score_n",
                    "hidden_benchmark_total_valid",
                    "hidden_benchmark_total_valid_n",
                    "hidden_benchmark_undetected",
                    "hidden_benchmark_undetected_n",
                    "full_pre_score",
                    "full_pre_n",
                    "delta_feedback",
                ]},
            }
            writer.writerow({k: out.get(k) for k in fields})


def main():
    rows, hidden_runs = collect_rows()
    os.makedirs(TABLES_DIR, exist_ok=True)
    scores_path = os.path.join(TABLES_DIR, "tab-gpt5-agent-hidden-ablation-scores.tex")
    csv_path = os.path.join(TABLES_DIR, "tab-gpt5-agent-hidden-ablation.csv")
    write_scores_table(rows, scores_path)
    write_csv(rows, csv_path)

    complete = [
        (run_id, lib)
        for run_id in HIDDEN_RUN_IDS
        for lib in residual.DOMAIN_MAP
        if hidden_runs[run_id][lib].get("complete")
    ]
    both_complete = [
        lib for lib in residual.DOMAIN_MAP
        if rows[lib].get("hidden_score_n") == len(HIDDEN_RUN_IDS)
    ]
    deltas = [
        rows[lib]["delta_feedback"]
        for lib in residual.DOMAIN_MAP
        if rows[lib].get("delta_feedback") is not None
    ]
    print(f"wrote {os.path.relpath(scores_path, SCRIPT_DIR)}")
    print(f"wrote {os.path.relpath(csv_path, SCRIPT_DIR)}")
    print(f"hidden complete cells: {len(complete)}/{len(HIDDEN_RUN_IDS) * len(residual.DOMAIN_MAP)}")
    print(f"hidden libraries complete in both runs: {len(both_complete)}/{len(residual.DOMAIN_MAP)}")
    if deltas:
        print(f"mean Delta_feedback (pp): {sum(deltas) / len(deltas):.2f}")


if __name__ == "__main__":
    main()
