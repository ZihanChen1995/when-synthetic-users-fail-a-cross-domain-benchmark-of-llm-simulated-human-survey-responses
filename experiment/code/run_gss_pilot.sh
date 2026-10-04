#!/usr/bin/env bash
# Stage 2 inference, GSS: Claude Haiku 4.5 and Sonnet 4.6 (styles A and C, seeds 0 and 1).
#
# Set HAIKU_MODEL_ID / SONNET_MODEL_ID / LLAMA8B_MODEL_ID / LLAMA70B_MODEL_ID as needed
# and implement APIClient.generate() in inference.py. Runs are cache-resumable.
# Not needed to reproduce the paper: all model outputs are cached.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-../../.venv/bin/python}"
LOG=logs
mkdir -p "$LOG"

# model_id|short_name pairs
MODELS=(
  "${HAIKU_MODEL_ID:?set this model id}|claude-haiku-4.5"
  "${SONNET_MODEL_ID:?set this model id}|claude-sonnet-4.6"
)

for entry in "${MODELS[@]}"; do
  mid="${entry%%|*}"; name="${entry##*|}"
  echo "=== $(date -u +%FT%TZ) starting $name ===" >> "$LOG/pilot.log"
  $PY run_inference.py --backend api --model "$mid" --model-name "$name" \
      --styles A C --seeds 0 1 >> "$LOG/${name}.log" 2>&1
  echo "=== $(date -u +%FT%TZ) finished $name ===" >> "$LOG/pilot.log"
done
echo "=== $(date -u +%FT%TZ) ALL MODELS DONE ===" >> "$LOG/pilot.log"
