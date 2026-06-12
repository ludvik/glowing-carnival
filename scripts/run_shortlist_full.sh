#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_IDS=$(python3 - <<'PY'
import json

catalog = json.load(open("config/model_catalog.json", encoding="utf-8"))
print(",".join(catalog["models"].keys()))
PY
)

uv run python scripts/run_eval.py \
  --dataset data/labels/classification_corpus.jsonl \
  --model-catalog config/model_catalog.json \
  --prompt config/prompts/classification_template.txt \
  --models "$MODEL_IDS" \
  --output-dir runs \
  --run-id "${1:-shortlist-full-v1}" \
  --concurrency "${EVAL_CONCURRENCY:-2}" \
  --timeout-seconds 45 \
  --max-retries 1 \
  --temperature 0 \
  --max-output-tokens 2048 \
  --progress-interval 5 \
  --all
