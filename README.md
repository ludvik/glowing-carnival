# Issue Classification Model Evaluator

This repo contains a runnable evaluation app for comparing 
Serverless Inference models on GitHub issue classification.

The submitted package includes the application code, Docker setup, labeled
dataset artifacts, model metadata, and persisted evaluation runs. The UI is
designed to load existing results first, so reviewers can inspect the evaluation
without making paid inference calls.

## Quick Start

Start the app with Docker Compose:

```bash
docker compose up --build
```

Open:

```text
http://localhost:8501
```

The Compose setup mounts the repo's local `./runs` directory into the container
at `/app/runs`, so the checked-in evaluation runs are available immediately.
No API key is required to browse existing results.

In the UI:

1. Choose a resultset from the left navigation.
2. Compare two models, or open the model landscape view.
3. Inspect scored quality, side-by-side failures, unscored behavior, operational
   metrics, and dataset examples.

## Key Deliverables

- Streamlit UI: `app.py`
- Evaluation harness: `src/eval_harness/`
- CLI runner: `scripts/run_eval.py`
- Docker setup: `Dockerfile`, `docker-compose.yml`
- Labeled corpus:
  - `data/labels/classification_corpus.jsonl`
  - `data/labels/scored_set.csv`
  - `data/labels/scored_set.jsonl`
  - `data/labels/unscored_set.csv`
  - `data/labels/labeling_summary.json`
- Model inventory and pricing metadata:
  - `data/models/`
  - `config/model_metadata.json`
  - `config/model_catalog.json`
- Persisted evaluation runs: `runs/`
- Review rationale/report:
  - `RATIONAL.md`
  - `docs/RATIONAL.pdf`

## API Keys

Existing results can be browsed without credentials.

To launch new model calls from the UI or CLI, export one of:

```bash
export DIGITALOCEAN_SI_API_KEY=<your-key>
```

The app also accepts `DO_INFERENCE_API_KEY` and `DIGITALOCEAN_TOKEN` for
compatibility.

## Dataset

The full corpus is `data/labels/classification_corpus.jsonl`. It contains every
doctl issue exactly once, with a `split` of `scored`, `unscored`, or `review`.

Only the certified scored subset is used for accuracy, F1, recall, and confusion
matrices. Unscored issues are still classified by models and shown in the UI for
agreement and behavior analysis.

Current scored support:

| Label | Count |
| --- | ---: |
| bug | 35 |
| enhancement | 35 |
| security | 26 |
| question | 18 |
| documentation | 10 |
| other | 2 |

`other` is intentionally thin and should not dominate model selection.

To rebuild the scored artifacts:

```bash
uv run python scripts/build_scored_set.py \
  --input data/doctl_issues.json \
  --output-dir data/labels \
  --min-confidence 0.75 \
  --seed 42 \
  --max-per-class 35
```

## Running New Evaluations

The UI can browse persisted runs without calling APIs. To run a new full-corpus
comparison from the CLI:

```bash
uv run python scripts/run_eval.py \
  --dataset data/labels/classification_corpus.jsonl \
  --model-catalog config/model_catalog.json \
  --prompt config/prompts/classification_template.txt \
  --models mistral-3-14B,openai-gpt-oss-20b \
  --output-dir runs \
  --run-id my-comparison \
  --concurrency 2 \
  --timeout-seconds 45 \
  --max-retries 1 \
  --temperature 0 \
  --max-output-tokens 2048 \
  --all
```

Run artifacts are written to:

```text
runs/{run_id}/
  run.json
  progress/{model_id}.json
  results/{model_id}.json
```

Each resultset records per-call latency, usage, cost, response headers, retries,
raw model output, parsed label, rationale, and errors.

## Regenerating Model Metadata

The model inventory snapshot is produced by:

```bash
python3 scripts/build_model_inventory.py \
  --output-dir data/models \
  --metadata config/model_metadata.json
```

`config/model_metadata.json` enriches selected models with local pricing and
capability metadata because the `/v1/models` endpoint does not include token
prices.

## Notes

- The UI should be the main review surface.
- `docs/RATIONAL.pdf` contains the condensed evaluation narrative and final
  recommendation.
- Persisted runs are included so reviewers do not need to spend API credits to
  inspect the result.
