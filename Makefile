# Thesis table/figure pipeline for the GPT-5.4 mutation-testing study.
#
# Prerequisites:
#   - the run data unpacked at artifact/data/  (fetch from the archive DOI; see README)
#   - a Python environment with artifact/requirements.txt installed
#     (override the interpreter with `make PY=/path/to/python <target>`)
#
# `make all` regenerates every table and figure. Figures read the CSV
# intermediates (and, for fig-comparison, the .tex tables) that the table
# targets emit, so tables run first; keep serial (.NOTPARALLEL below).

PY ?= python3
RUN = cd artifact && $(PY)
DIST := dist

.NOTPARALLEL:
.PHONY: all tables figures dist clean-dist \
        comparison-tables token-tables agent-comparison temperature residual-cost \
        correctness hidden-csv nontriviality-csv nontriviality \
        fig-comparison fig-scatter fig-temperature fig-hidden fig-token \
        fig-agent-testpilot fig-residual

all: tables figures

# ---------------------------------------------------------------- tables ----
# Each target runs one generator; outputs land in artifact/tables/.

comparison-tables:   # tab-gpt5-testpilot-full-metrics, tab-gpt5-mutation-comparison, tab-gpt5-agent-general-mutation-score (+ unconsumed extras)
	$(RUN) generate_gpt5_comparison_tables.py

token-tables:        # tab-gpt5-token-comparison, tab-gpt5-testpilot-token-comparison, tab-gpt5-token-usage.csv
	$(RUN) generate_gpt5_token_table.py

agent-comparison:    # tab-gpt5-agent-testpilot-comparison, tab-gpt5-agent-testpilot-per-library (+ .csv)
	$(RUN) generate_gpt5_agent_testpilot_comparison_table.py

temperature:         # tab-gpt5-agent-residual-scores, tab-gpt5-agent-temperature.csv
	$(RUN) generate_gpt5_agent_temperature_table.py

residual-cost:       # tab-gpt5-agent-residual-cost (+ tab-gpt5-agent-residual.csv)
	$(RUN) generate_gpt5_agent_residual_table.py

correctness:         # tab-gpt5-correctness-failure-overview (+ .csv)
	$(RUN) generate_gpt5_correctness_failure_table.py

hidden-csv:          # tab-gpt5-agent-hidden-ablation.csv (+ tab-gpt5-agent-hidden-ablation-scores.tex, unconsumed)
	$(RUN) generate_gpt5_agent_hidden_ablation_table.py

nontriviality-csv:   # tab-gpt5-agent-nontriviality.csv + tab-gpt5-agent-nontriviality-tests.csv
	$(RUN) generate_gpt5_agent_nontriviality.py

nontriviality: nontriviality-csv   # tab-nontriviality-testpilot-vs-agent
	$(RUN) generate_nontriviality_table.py

tables: comparison-tables token-tables agent-comparison temperature \
        residual-cost correctness hidden-csv nontriviality

# --------------------------------------------------------------- figures ----
# Outputs land in artifact/figures/ as .pdf + .png.

fig-comparison:      # gpt5_testpilot_comparison_dumbbell  (parses the comparison .tex tables)
	$(RUN) generate_gpt5_comparison_figure.py

fig-scatter:         # gpt5_agent_coverage_vs_mutation_scatter
	$(RUN) generate_gpt5_agent_scatter.py

fig-temperature:     # gpt5_temperature_dumbbell  (reads tab-gpt5-agent-temperature.csv)
	$(RUN) generate_gpt5_temperature_figure.py

fig-hidden:          # gpt5_hidden_ablation_dumbbell  (reads tab-gpt5-agent-hidden-ablation.csv)
	$(RUN) generate_gpt5_hidden_ablation_figure.py

fig-token:           # gpt5_token_scaling_scatter  (reads tab-gpt5-token-usage.csv)
	$(RUN) generate_gpt5_token_figure.py

fig-agent-testpilot: # gpt5_agent_testpilot_dumbbell  (reads tab-gpt5-agent-testpilot-per-library.csv)
	$(RUN) generate_gpt5_agent_testpilot_figure.py

fig-residual:        # gpt5_residual_dumbbell  (reads tab-gpt5-agent-temperature.csv)
	$(RUN) generate_gpt5_residual_figure.py

figures: fig-comparison fig-scatter fig-temperature fig-hidden fig-token \
         fig-agent-testpilot fig-residual

# ------------------------------------------------------------------ dist ----
# Stage the thesis publish set (11 tables + 7 figures) in dist/, mirroring the
# Overleaf directory layout, ready for upload.

dist:
	rm -rf $(DIST)
	mkdir -p $(DIST)/tables/final $(DIST)/tables/testpilot $(DIST)/tables/pilot \
	         $(DIST)/tables/misc $(DIST)/figures
	cp artifact/tables/tab-gpt5-agent-residual-cost.tex \
	   artifact/tables/tab-gpt5-testpilot-full-metrics.tex \
	   artifact/tables/tab-gpt5-agent-testpilot-per-library.tex \
	   artifact/tables/tab-gpt5-agent-residual-scores.tex $(DIST)/tables/final/
	cp artifact/tables/tab-gpt5-testpilot-token-comparison.tex $(DIST)/tables/testpilot/
	cp artifact/tables/tab-gpt5-mutation-comparison.tex \
	   artifact/tables/tab-gpt5-agent-general-mutation-score.tex \
	   artifact/tables/tab-gpt5-token-comparison.tex $(DIST)/tables/pilot/
	cp artifact/tables/tab-gpt5-correctness-failure-overview.tex \
	   artifact/tables/tab-nontriviality-testpilot-vs-agent.tex $(DIST)/tables/misc/
	cp artifact/tables/tab-gpt5-agent-testpilot-comparison.tex $(DIST)/tables/
	cp artifact/figures/gpt5_testpilot_comparison_dumbbell.pdf \
	   artifact/figures/gpt5_agent_coverage_vs_mutation_scatter.pdf \
	   artifact/figures/gpt5_temperature_dumbbell.pdf \
	   artifact/figures/gpt5_hidden_ablation_dumbbell.pdf \
	   artifact/figures/gpt5_token_scaling_scatter.pdf \
	   artifact/figures/gpt5_agent_testpilot_dumbbell.pdf \
	   artifact/figures/gpt5_residual_dumbbell.pdf $(DIST)/figures/
	@echo "dist/ staged (Overleaf layout)."

clean-dist:
	rm -rf $(DIST)
