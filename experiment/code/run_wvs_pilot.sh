#!/usr/bin/env bash
# Stage 2 inference, WVS: all four models (styles A and C, seeds 0 and 1).
#
# Set HAIKU_MODEL_ID / SONNET_MODEL_ID / LLAMA8B_MODEL_ID / LLAMA70B_MODEL_ID as needed
# and implement APIClient.generate() in inference.py. Runs are cache-resumable.
# Not needed to reproduce the paper: all model outputs are cached.
set -euo pipefail
cd "$(dirname "$0")"
export LLM_FAULTS_DATASET=WVS
PY="${PYTHON:-../../.venv/bin/python}"
LOG=logs
mkdir -p "$LOG"

MODELS=(
  "${HAIKU_MODEL_ID:?set this model id}|claude-haiku-4.5"
  "${SONNET_MODEL_ID:?set this model id}|claude-sonnet-4.6"
  "${LLAMA8B_MODEL_ID:?set this model id}|llama3.1-8b"
  "${LLAMA70B_MODEL_ID:?set this model id}|llama3.3-70b"
)
for entry in "${MODELS[@]}"; do
  mid="${entry%%|*}"; name="${entry##*|}"
  echo "=== $(date -u +%FT%TZ) WVS starting $name ===" >> "$LOG/wvs_pilot.log"
  $PY run_inference.py --backend api --model "$mid" --model-name "$name" \
      --styles A C --seeds 0 1 >> "$LOG/wvs_${name}.log" 2>&1
  echo "=== $(date -u +%FT%TZ) WVS finished $name ===" >> "$LOG/wvs_pilot.log"
done
echo "=== $(date -u +%FT%TZ) WVS ALL MODELS DONE ===" >> "$LOG/wvs_pilot.log"
