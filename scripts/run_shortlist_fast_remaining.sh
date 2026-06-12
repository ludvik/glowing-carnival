#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Generated from the current run plan:
# - skip models that already have a 530-issue resultset
# - skip models with auth/no-output failures in screening
# - skip models projected to take more than 10 minutes for 530 issues
# - run router-policy candidates separately because they do not yet have full results
MODEL_IDS="router:general,router:knowledge-base-document,router:software-engineering,router:writing"

uv run python scripts/run_eval.py \
  --dataset data/labels/classification_corpus.jsonl \
  --model-catalog config/model_catalog.json \
  --prompt config/prompts/classification_template.txt \
  --models "$MODEL_IDS" \
  --output-dir runs \
  --run-id "${1:-shortlist-router-full-v1}" \
  --concurrency "${EVAL_CONCURRENCY:-2}" \
  --timeout-seconds 45 \
  --max-retries 1 \
  --temperature 0 \
  --max-output-tokens 2048 \
  --progress-interval 5 \
  --all
