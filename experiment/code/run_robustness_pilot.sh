#!/usr/bin/env bash
# Stage 2 inference, RQ4 perturbations. GSS: option order + persona for the two
# Claude models (2 seeds). WVS: option-order spot-check (1 seed).
#
# Set HAIKU_MODEL_ID / SONNET_MODEL_ID / LLAMA8B_MODEL_ID / LLAMA70B_MODEL_ID as needed
# and implement APIClient.generate() in inference.py. Runs are cache-resumable.
# Not needed to reproduce the paper: all model outputs are cached.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-../../.venv/bin/python}"
LOG=logs
mkdir -p "$LOG"

CLOSED=(
  "${HAIKU_MODEL_ID:?set this model id}|claude-haiku-4.5"
  "${SONNET_MODEL_ID:?set this model id}|claude-sonnet-4.6"
)

echo "=== $(date -u +%FT%TZ) ROBUSTNESS start ===" >> "$LOG/robustness.log"

# GSS: option-order + persona, both closed models, 2 seeds
export LLM_FAULTS_DATASET=GSS
for entry in "${CLOSED[@]}"; do
  mid="${entry%%|*}"; name="${entry##*|}"
  echo "=== $(date -u +%FT%TZ) GSS $name order+persona ===" >> "$LOG/robustness.log"
  $PY run_robustness.py --model "$mid" --model-name "$name" \
      --variants order persona --seeds 0 1 >> "$LOG/rob_gss_${name}.log" 2>&1
done

# WVS: option-order spot-check, both closed models, 1 seed
export LLM_FAULTS_DATASET=WVS
for entry in "${CLOSED[@]}"; do
  mid="${entry%%|*}"; name="${entry##*|}"
  echo "=== $(date -u +%FT%TZ) WVS $name order ===" >> "$LOG/robustness.log"
  $PY run_robustness.py --model "$mid" --model-name "$name" \
      --variants order --seeds 0 >> "$LOG/rob_wvs_${name}.log" 2>&1
done

echo "=== $(date -u +%FT%TZ) ROBUSTNESS DONE ===" >> "$LOG/robustness.log"
