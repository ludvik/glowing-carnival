# glowing-carnival

Evaluation harness for the DigitalOcean FDE exercise.

## Fetch the issue corpus

This project uses `uv` for Python environment management.

```bash
uv run python scripts/fetch_github_issues.py
```

The script writes a stable JSON corpus to `data/doctl_issues.json`. It uses the
GitHub public API and can optionally use `GITHUB_TOKEN` to raise rate limits.

## Build the golden dataset

```bash
uv run python scripts/build_golden_dataset.py
```

The script writes `data/golden_dataset.json`, which splits the 530 fetched issues
into:

- `scored`: issues with a ground-truth label for evaluation.
- `unscored`: issues that still get model predictions, but are excluded from
  accuracy/F1 scoring.
- `needs_review`: conflicting cases not yet manually adjudicated.

Ground truth is derived from maintainer labels when exactly one target class can
be mapped without ambiguity:

- `bug` -> `bug`
- `suggestion`, `enhancement`, `api-parity` -> `enhancement`
- `question`, `troubleshooting` -> `question`
- `docs` -> `documentation`
- `security vulnerability` -> `security`
- `duplicate` -> `other`

Workflow and metadata labels such as `hacktoberfest`, `packaging`, `snap`,
`waiting-response`, and `Needs Investigation` are ignored for scoring.

Conflicting mapped labels are not scored automatically. The current dataset uses
`data/golden_overrides.json` for six manual adjudications, each with a rationale.
That produces 306 scored issues and 224 unscored issues:

| Label | Count |
| --- | ---: |
| bug | 158 |
| enhancement | 104 |
| security | 26 |
| question | 15 |
| documentation | 2 |
| other | 1 |

The `documentation` and `other` classes are intentionally retained, but their
sample sizes are too small for strong per-class claims. Treat their metrics as
qualitative signals unless more labels are added.

## Run the eval engine

The eval engine calls DigitalOcean Serverless Inference through its
OpenAI-compatible chat completions API. Set a model access key or DigitalOcean
token before running:

```bash
export DIGITALOCEAN_SI_API_KEY=...
```

Run a cost-controlled smoke eval:

```bash
uv run python scripts/run_eval.py \
  --prompt config/prompts/classification_template.txt \
  --models mistral-3-14b,gpt-oss-20b \
  --limit 3 \
  --concurrency 2
```

Run the full corpus explicitly:

```bash
uv run python scripts/run_eval.py \
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

`runs/` is gitignored. Use `EVAL_OUTPUT_DIR` or `--output-dir` to write results
to a mounted volume in deployed environments.

Retry retryable failed calls from a prior model resultset:

```bash
uv run python scripts/run_eval.py \
  --retry-failed runs/{run_id}/results/{model_id}.json
```

Run a small concurrency sweep before choosing a production default:

```bash
uv run python scripts/sweep_concurrency.py \
  --model mistral-3-14b \
  --limit 20 \
  --concurrency-values 1,2,4,8
```

Each model resultset includes wall-clock time and sustained throughput in
`operational_summary.throughput`.

## Score persisted resultsets

Generate the scored, unscored, and operational JSON summaries required by the
eval UI:

```bash
uv run python scripts/score_results.py \
  --resultsets runs/{run_id}/results/{model_a}.json,runs/{run_id}/results/{model_b}.json
```

The script writes:

```text
runs/{run_id}/summaries/
  scored_metrics.json
  unscored_analysis.json
  operational_metrics.json
```

`scored_metrics.json` includes accuracy, per-class precision/recall/F1,
confusion matrices, cost per correct classification, and scored model
disagreements with ground truth. `unscored_analysis.json` includes model
agreement, per-class prediction distributions, raw outputs, and disagreements
for issues without ground truth.
