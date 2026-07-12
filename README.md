# GPT-5.4 Mutation-Testing Study — Analysis Pipeline & Generated Artifacts

Scripts and generated tables/figures for a thesis evaluating LLM-based JavaScript
unit-test generation with mutation analysis: replications of
[TestPilot](https://github.com/githubnext/testpilot) with GPT-5.4 (two output-token
budgets) and a custom mutation-guided agentic test generator, scored with
[StrykerJS](https://stryker-mutator.io/) across a 20-library benchmark corpus.

This repository is one of three publication pieces:

1. **The agent**: [`agentic-stryker`](https://github.com/eefscheef/agentic-stryker) —
   the agentic test generator and benchmark harness. All thesis runs used
   development-history commit `63be77f`, recorded in each archived run's
   `manifest.json`.
2. **This data analysis repository**: the analysis pipeline that turns run data into every
   generated table and figure in the thesis (11 tables + 7 figures).
3. **The data archive**: [doi:10.5281/zenodo.21326300](https://doi.org/10.5281/zenodo.21326300) — the run data (`artifact/data/` tree,
   ≈7 GB): TestPilot-format runs augmented with Stryker mutation reports, and the
   full agent run trees including LLM transcripts.

## Provenance

The classic-model data (Codex cushman, GPT-3.5-turbo, StarCoder) derives from the
TestPilot artifact by Schäfer et al. (*An Empirical Evaluation of Using Large
Language Models for Automated Unit Test Generation*, IEEE TSE 2024):
[doi:10.6084/m9.figshare.23653371](https://doi.org/10.6084/m9.figshare.23653371)
(CC BY 4.0). The archive redistributes pruned copies of those runs augmented with
our mutation analysis (produced by
[eefscheef/mutation-pilot](https://github.com/eefscheef/mutation-pilot));
everything we pruned (per-test coverage data, Nessie data, the original authors'
stats) is recoverable from that DOI. The GPT-5.4 TestPilot runs were produced
with our TestPilot fork,
[eefscheef/testpilot2](https://github.com/eefscheef/testpilot2).

All mutation reports in the archive are the **post-idempotency-correction** state:
non-idempotent generated tests (which break Stryker's sandbox reuse and inflate
scores to a spurious ~100%) are excluded from mutation runs via a dry-run
admissibility filter. See the thesis for details.

Each agent run directory ships a `manifest.json` recording the exact run ID, agent
git SHA, condition flags, library set, and any splices.

## Layout

```
Makefile                    regenerate / stage everything (see below)
artifact/
  generate_gpt5_*.py        table + figure generators (the thesis pipeline)
  generate_nontriviality_table.py
  dumbbell_summary.py       shared dumbbell-plot styling (medians + caret means)
  analyze_residual_token_overhead.py
  pricing.json              API prices used for the cost columns
  requirements.txt          Python dependencies
  ql/                       CodeQL pack for the non-triviality analysis (optional:
                            its result CSVs are committed under tables/)
  tables/                   generated .tex tables + .csv intermediates (committed)
  figures/                  generated figures, .pdf + .png (committed)
  data/                     NOT in git — fetch from the archive DOI
```

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r artifact/requirements.txt
```

## Data

Download the archive from [doi:10.5281/zenodo.21326300](https://doi.org/10.5281/zenodo.21326300) (or run `./fetch-data.sh`) and unpack it so the per-model trees sit
at `artifact/data/<model>/<run>/<package>/` (the archive README maps each tarball to
this layout). Models: `cushman`, `gptturbo`, `starcoder` (10 runs each),
`gpt-5.4-output-500`, `gpt-5.4-output-1000` (1 run each), `gpt-5.4-agent` (pilot),
`gpt-5.4-agent-residual`, `gpt-5.4-agent-hidden`.

## Usage

```sh
make all PY=.venv/bin/python    # regenerate every table and figure (needs data/)
make dist                       # stage the publish set in dist/ (Overleaf layout)
```

Individual targets exist per generator (`make temperature`, `make fig-residual`, …);
see the Makefile. The generated outputs are committed, so `make all` after a fresh
clone + data fetch should reproduce them; generators print validation summaries
(means/medians) to cross-check against the thesis.

| Generator | Emits | Thesis location |
|---|---|---|
| `generate_gpt5_comparison_tables.py` | full-metrics, pilot mutation-comparison, pilot agent-scores tables | App. D §1, pilot appendix |
| `generate_gpt5_token_table.py` | both token tables + token CSV | App. D §2, pilot appendix |
| `generate_gpt5_agent_testpilot_comparison_table.py` | aggregate + per-library comparison (+ CSV) | results ch., App. D §3 |
| `generate_gpt5_agent_temperature_table.py` | residual-scores table + temperature CSV | App. D §4 |
| `generate_gpt5_agent_residual_table.py` | cost table | results ch. |
| `generate_gpt5_correctness_failure_table.py` | correctness table | results ch. |
| `generate_gpt5_agent_nontriviality.py` + `generate_nontriviality_table.py` | non-triviality CSVs + table | results ch. |
| `generate_gpt5_agent_hidden_ablation_table.py` | hidden-ablation CSV | (feeds figure) |
| `generate_gpt5_comparison_figure.py` … `generate_gpt5_residual_figure.py` | the 7 thesis figures | results ch. |

`generate_classic_mutation_comparison_table.py` is included as a supplementary
example consumer of the mutation-augmented classic-model data.

## CodeQL (optional)

The agent-side non-triviality numbers come from the CodeQL pack in `artifact/ql/`.
Its result CSVs (`tables/tab-gpt5-agent-nontriviality*.csv`) are committed, so the
pipeline runs without CodeQL; rebuild the databases only to reproduce those CSVs
from scratch.

## License & citation

Code: MIT (see `LICENSE`). The generated tables/figures and the archived data
derive in part from the TestPilot artifact (CC BY 4.0). Cite
[doi:10.6084/m9.figshare.23653371](https://doi.org/10.6084/m9.figshare.23653371)
when reusing the classic-model data.

Thesis citation and archive DOI: *TBD*.
