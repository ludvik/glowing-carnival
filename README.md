# Issue Classification Model Evaluator

This project evaluates DigitalOcean Serverless Inference models on GitHub issue
classification. It includes a curated issue corpus, labeled scored subset,
persisted evaluation runs, and a Streamlit UI for comparing two models side by
side.

## Quick Start

The fastest way to review the project is Docker Compose:

```bash
docker compose up --build
```

Then open:

```text
http://localhost:8501
```

The submitted project includes curated persisted run artifacts under `runs/`.
Docker Compose mounts the local `./runs` directory into the container at
`/app/runs`, so the UI can load the existing evaluations immediately. You do not
need an API key just to browse existing results.

In the UI:

1. Select an existing run.
2. Pick Model A and Model B.
3. Use the tabs to inspect overview metrics, side-by-side errors, scored
   quality, unscored behavior, operational metrics, and the underlying dataset.

The app is persisted-results-first. It does not call paid model APIs unless you
explicitly start a new comparison from the UI or run the eval scripts yourself.

If you want to launch new model calls, provide a DigitalOcean Serverless
Inference key before starting Compose:

```bash
export DIGITALOCEAN_SI_API_KEY=<DIGITALOCEAN_SI_API_KEY>
docker compose up --build
```

Compose passes these environment variables into the app if they are set:

- `DIGITALOCEAN_SI_API_KEY`
- `DO_INFERENCE_API_KEY`
- `DIGITALOCEAN_TOKEN`

## What Is Included

- `data/labels/classification_corpus.jsonl`: every doctl issue exactly once.
- `data/labels/scored_set.csv` and `scored_set.jsonl`: high-confidence scored
  subset used for accuracy/F1.
- `config/model_catalog.json`: 16 curated model candidates with pricing metadata.
- `runs/`: persisted model evaluation runs for the UI.
- `app.py`: Streamlit comparison UI.
- `scripts/run_eval.py`: CLI evaluation engine.

## Fetch the issue corpus

This project uses `uv` for Python environment management.

```bash
uv run python scripts/fetch_github_issues.py
```

The script writes a stable JSON corpus to `data/doctl_issues.json`. It uses the
GitHub public API and can optionally use `GITHUB_TOKEN` to raise rate limits.

## Build the scored dataset

```bash
uv run python scripts/build_scored_set.py \
  --input data/doctl_issues.json \
  --output-dir data/labels \
  --min-confidence 0.75 \
  --seed 42 \
  --max-per-class 35
```

The script treats maintainer labels as weak signals, combines them with
deterministic text heuristics, and writes the label artifacts under
`data/labels/`:

- `classification_corpus.jsonl`: all 530 input issues exactly once. This is the
  default model-classification input.
- `scored_set.csv` and `scored_set.jsonl`: high-confidence issues with one
  certified ground-truth label.
- `unscored_set.csv`: issues still classified by models but excluded from
  accuracy/F1 scoring.
- `review_queue.csv`: ambiguous cases that need adjudication.
- `manual_review_candidates.csv`: prioritized rows to inspect next, including
  review rows and thin-label candidates.
- `labeling_summary.json`: counts, support by label, thin labels, and exclusion
  reasons.
- `labeling_guide.md`: the documented labeling methodology.

Manual corrections live in `data/labels/manual_overrides.csv`. Overrides take
precedence over automatic rules and must include a rationale. The checked-in v1
scored set is intentionally conservative:

| Label | Count |
| --- | ---: |
| bug | 35 |
| enhancement | 35 |
| security | 26 |
| question | 18 |
| documentation | 14 |
| other | 2 |

`other` remains a thin class. Treat its metrics as qualitative unless more
manual labels are added.


## Run the Eval Engine

The eval engine calls DigitalOcean Serverless Inference through its
OpenAI-compatible chat completions API. Set a model access key or DigitalOcean
token before running:

```bash
export DIGITALOCEAN_SI_API_KEY=<DIGITALOCEAN_SI_API_KEY>
```

Run a small screening eval:

```bash
uv run python scripts/run_eval.py \
  --dataset data/labels/classification_corpus.jsonl \
  --prompt config/prompts/classification_template.txt \
  --models mistral-3-14b,gpt-oss-20b \
  --limit 3 \
  --concurrency 2
```

Run the full corpus explicitly:

```bash
uv run python scripts/run_eval.py \
  --dataset data/labels/classification_corpus.jsonl \
  --prompt config/prompts/classification_template.txt \
  --models mistral-3-14b,gpt-oss-20b \
  --all \
  --concurrency 8
```

The engine writes runtime artifacts to `runs/{run_id}/` by default:

```text
runs/{run_id}/
  run.json
  progress/{model_id}.json
  results/{model_id}.json
```

During long runs, `progress/{model_id}.json` is updated every
`--progress-interval` completed calls and includes completed count, ok/error
count, current requests/sec, and ETA.

Runtime artifacts are written under `runs/`. For the review package, curated
persisted runs should be included so the UI can be opened and inspected without
making new paid inference calls. Use `EVAL_OUTPUT_DIR` or `--output-dir` to
write results to another mounted volume in deployed environments.

Retry retryable failed calls from a prior model resultset:

```bash
uv run python scripts/run_eval.py \
  --retry-failed runs/{run_id}/results/{model_id}.json
```

Each model resultset includes wall-clock time and sustained throughput in
`operational_summary.throughput`.

The UI reads persisted resultsets directly from `runs/{run_id}/results/` and
computes scored, unscored, and operational views without an extra summary step.

## Run the UI Locally

Run locally:

```bash
uv run streamlit run app.py
```

Or run with Docker Compose:

```bash
export DIGITALOCEAN_SI_API_KEY=...
docker compose up --build
```

The default Compose setup mounts the repo's local `./runs` directory into
`/app/runs`, so existing persisted run artifacts are visible in the UI and new
runtime artifacts survive image rebuilds.

To run the image manually with the local `runs/` directory:

```bash
docker build -t model-eval .
docker run --rm -p 8501:8501 \
  -v "$PWD/runs:/app/runs" \
  model-eval
```

The UI reads persisted run artifacts only. It does not call paid inference APIs
unless a separate runner command is executed.
