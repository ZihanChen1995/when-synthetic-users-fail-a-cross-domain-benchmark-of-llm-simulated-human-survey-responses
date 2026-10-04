# When Synthetic Users Fail -- reproduction targets.
#
#   make setup      create .venv and install pinned dependencies
#   make smoke      ~30 s   verify this checkout is complete
#   make verify     <1 min  regenerate the main table, diff against the paper
#   make reproduce  30-60 min  regenerate every number and figure in the paper
#   make clean      remove build artifacts (never touches data)

PYTHON ?= python3.12
VENV   := .venv
PY     := $(VENV)/bin/python
CODE   := experiment/code
DOMAINS := GSS WVS

# analysis scripts, in dependency order. summarize.py also recomputes the per-model
# metrics and subgroup files. analyze_weights.py is excluded: it re-reads the raw
# survey files (see docs/RAW_DATA.md).
SCRIPTS := summarize.py analyze_rq1_robustness.py analyze_invalid_outputs.py \
           analyze_supplementary.py analyze_decision_impact.py analyze_robustness.py \
           analyze_appendix_robustness.py make_ci_figs.py make_decision_fig.py

# analyze_rq1_robustness.py needs the full WVS table, which the WVSA terms do not
# allow us to redistribute -- it runs on GSS and is skipped on WVS unless the user
# has rebuilt that table from their own download.
SKIP_WVS := analyze_rq1_robustness.py

.PHONY: help setup smoke verify reproduce weights clean

help:
	@sed -n '3,7p' Makefile | sed 's/^# \{0,1\}//'

setup:
	@echo ">> creating $(VENV) with $(PYTHON)"
	@$(PYTHON) -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)' || \
	  (echo "ERROR: $(PYTHON) is too old; need >= 3.11. Set PYTHON=python3.12 or similar." && exit 1)
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	@echo ">> done. Next: make smoke"

smoke:
	$(PY) scripts/smoke_test.py

verify:
	$(PY) scripts/verify.py

reproduce:
	@echo ">> regenerating all analysis outputs for: $(DOMAINS)"
	@for ds in $(DOMAINS); do \
	  for s in $(SCRIPTS); do \
	    if [ "$$ds" = "WVS" ] && echo " $(SKIP_WVS) " | grep -q " $$s "; then \
	      echo ">> [$$ds] $$s -- SKIPPED (needs WVS microdata; see docs/RAW_DATA.md)"; \
	      continue; \
	    fi; \
	    echo ">> [$$ds] $$s"; \
	    (cd $(CODE) && LLM_FAULTS_DATASET=$$ds ../../$(PY) $$s) || exit 1; \
	  done; \
	done
	@echo ">> done. Outputs under experiment/analysis/{GSS,WVS}/"
	@echo ">> now run: make verify"

# survey-weight robustness appendix. Needs the raw GSS/WVS files -- see docs/RAW_DATA.md.
weights:
	@for ds in $(DOMAINS); do \
	  echo ">> [$$ds] analyze_weights.py"; \
	  (cd $(CODE) && LLM_FAULTS_DATASET=$$ds ../../$(PY) analyze_weights.py) || \
	    echo ">> [$$ds] SKIPPED -- raw survey data not present (expected; see docs/RAW_DATA.md)"; \
	done

clean:
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	@echo ">> cleaned (data and analysis outputs untouched)"
