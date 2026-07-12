#!/usr/bin/env python3
"""
Cross-model mutation-score comparison for the three classic TestPilot models
(Cushman, GPT-Turbo, StarCoder).

This is the mutation-score analogue of tab-llm-comparison (which compares
statement/branch coverage). It answers whether the coverage/passing trends
across the classic models also hold for mutation score.

Per-library scores are the MEDIAN across each model's runs, computed by the same
loaders that feed the GPT-Turbo column of the GPT-5.4 comparison tables
(generate_gpt5_comparison_tables.collect_mutation_scores), so the numbers tie
out. Libraries without a usable mutation baseline are excluded via the shared
NO_MUTATION_BASELINE_LIBS set (Node-incompatible libs plus the fs-extra
environment-overfitting case), and a run with no passing tests counts as 0.0%
(matching the other mutation tables). The per-model Median row is the median
across the per-library scores shown.

Outputs:
  tables/tab-classic-mutation-comparison.tex
  tables/tab-classic-mutation-comparison.csv
"""

import csv
import os

import generate_gpt5_comparison_tables as comparison

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR = os.path.join(SCRIPT_DIR, "tables")

MODELS = ["cushman", "gptturbo", "starcoder"]
MODEL_DISPLAY_NAMES = {
    "cushman": "Cushman",
    "gptturbo": "GPT-Turbo",
    "starcoder": "StarCoder",
}

WARN = "%WARNING: auto-generated. Edit generate_classic_mutation_comparison_table.py instead."


def collect():
    """{model: {lib: score-or-None}} using the shared median-across-runs loaders."""
    scores = {}
    for model in MODELS:
        model_dir = os.path.join(comparison.DATA_DIR, model)
        mut = comparison.collect_mutation_scores(model_dir)
        cov = comparison.collect_coverage_stats(model_dir)
        scores[model] = {
            lib: comparison._mutation_score_or_zero_if_no_passing(mut, cov, lib)
            for lib in comparison.DOMAIN_MAP
            if lib not in comparison.NO_MUTATION_BASELINE_LIBS
        }
    return scores


def _libs():
    return [
        lib
        for lib in comparison.DOMAIN_MAP
        if lib not in comparison.NO_MUTATION_BASELINE_LIBS
    ]


def fmt_cell(value, is_row_max):
    if value is None:
        return "--"
    formatted = f"{value:.1f}\\%"
    return f"\\textbf{{{formatted}}}" if is_row_max else formatted


def write_tex(scores, path):
    libs = _libs()
    col_spec = "l" + "r" * len(MODELS)
    lines = [
        WARN,
        r"\begin{table}[t!]",
        r"\centering",
        r"\caption{Mutation score per project for the three classic \testpilot "
        r"models. Scores are the median across each model's runs; the bold cell "
        r"in each row is the highest-scoring model. Runs with no passing tests "
        r"count as 0.0\%. The Median row is the median across the projects shown.}",
        r"\label{tab:classic-mutation-comparison}",
        r"\resizebox{0.7\columnwidth}{!}{",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        r"\textbf{Project} & "
        + " & ".join(rf"\textbf{{{MODEL_DISPLAY_NAMES[m]}}}" for m in MODELS)
        + r" \\",
        r"\midrule",
    ]

    for lib in libs:
        row_vals = [scores[m].get(lib) for m in MODELS]
        present = [v for v in row_vals if v is not None]
        row_max = max(present) if present else None
        cells = [
            fmt_cell(v, row_max is not None and v == row_max)
            for v in row_vals
        ]
        lines.append(f"{lib} & " + " & ".join(cells) + r" \\")
        if lib in comparison.SEPARATOR_LIBS:
            lines.append(r"\midrule")

    lines.append(r"\midrule")
    median_vals = [
        comparison._med([scores[m][lib] for lib in libs if scores[m].get(lib) is not None])
        for m in MODELS
    ]
    present = [v for v in median_vals if v is not None]
    med_max = max(present) if present else None
    median_cells = [
        fmt_cell(v, med_max is not None and v == med_max) for v in median_vals
    ]
    lines.append(r"\textbf{Median} & " + " & ".join(median_cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def write_csv(scores, path):
    libs = _libs()
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["project"] + MODELS)
        for lib in libs:
            writer.writerow([lib] + [scores[m].get(lib) for m in MODELS])
        writer.writerow(
            ["__median__"]
            + [
                comparison._med(
                    [scores[m][lib] for lib in libs if scores[m].get(lib) is not None]
                )
                for m in MODELS
            ]
        )


def main():
    scores = collect()
    os.makedirs(TABLES_DIR, exist_ok=True)
    tex_path = os.path.join(TABLES_DIR, "tab-classic-mutation-comparison.tex")
    csv_path = os.path.join(TABLES_DIR, "tab-classic-mutation-comparison.csv")
    write_tex(scores, tex_path)
    write_csv(scores, csv_path)

    print(f"wrote {os.path.relpath(tex_path, SCRIPT_DIR)}")
    print(f"wrote {os.path.relpath(csv_path, SCRIPT_DIR)}")
    libs = _libs()
    for m in MODELS:
        vals = [scores[m][lib] for lib in libs if scores[m].get(lib) is not None]
        med = comparison._med(vals)
        print(f"{MODEL_DISPLAY_NAMES[m]:10s}: {len(vals)}/{len(libs)} libs scored, "
              f"median {med:.1f}%" if med is not None else f"{m}: no scores")


if __name__ == "__main__":
    main()
