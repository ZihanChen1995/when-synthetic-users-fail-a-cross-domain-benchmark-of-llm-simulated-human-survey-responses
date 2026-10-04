#!/usr/bin/env bash
# Stage 2 inference, GSS: Llama 3.1-8B and Llama 3.3-70B (styles A and C, seeds 0 and 1).
#
# Set HAIKU_MODEL_ID / SONNET_MODEL_ID / LLAMA8B_MODEL_ID / LLAMA70B_MODEL_ID as needed
# and implement APIClient.generate() in inference.py. Runs are cache-resumable.
# Not needed to reproduce the paper: all model outputs are cached.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-../../.venv/bin/python}"
LOG=logs
mkdir -p "$LOG"

MODELS=(
  "${LLAMA8B_MODEL_ID:?set this model id}|llama3.1-8b"
  "${LLAMA70B_MODEL_ID:?set this model id}|llama3.3-70b"
)
for entry in "${MODELS[@]}"; do
  mid="${entry%%|*}"; name="${entry##*|}"
  echo "=== $(date -u +%FT%TZ) starting $name ===" >> "$LOG/pilot.log"
  $PY run_inference.py --backend api --model "$mid" --model-name "$name" \
      --styles A C --seeds 0 1 >> "$LOG/${name}.log" 2>&1
  echo "=== $(date -u +%FT%TZ) finished $name ===" >> "$LOG/pilot.log"
done
echo "=== $(date -u +%FT%TZ) LLAMA MODELS DONE ===" >> "$LOG/pilot.log"
