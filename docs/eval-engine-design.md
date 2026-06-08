# Eval Engine Design

## Scope

The eval engine is a CLI-runnable execution layer for model evaluation. It reads
a dataset, runs candidate models against every selected issue, and persists one
independently inspectable result record for each issue-model call.

The engine is responsible for:

- Loading the effective dataset for a run.
- Loading candidate model and pricing configuration.
- Building the classification prompt for each issue.
- Sending one inference request per issue per model.
- Enforcing configurable concurrency.
- Recording per-call timing, usage, cost, raw output, parsed label, and errors.
- Persisting run metadata and model resultsets.
- Producing enough data for scoring, UI drill-down, retry, and cost tracing.

The engine is not responsible for:

- Deciding how ground truth is certified.
- Mutating the checked-in golden dataset during deployed runtime.
- Computing final model recommendations.
- Rendering the UI.
- Hiding failures from scoring or presentation.

The scoring layer reads engine outputs and ground truth to compute accuracy,
precision, recall, F1, confusion matrices, model agreement, class distribution,
latency percentiles, throughput, cost, and error rates.

## Operating Modes

The same CLI engine supports local development and deployed execution.

### Development / Curation Mode

Development mode is used to update artifacts that are intended to be committed to
the repository, such as the stable issue corpus, curated ground truth, effective
eval dataset, and selected baseline runs.

Characteristics:

- Runs locally through `uv` or Python.
- Writes curated data under `data/`.
- Changes are reviewed with `git diff` before commit.
- Used to promote trusted annotation changes into the repository.

### Deployed / Runtime Mode

Deployed mode is used by the app and reviewers after the container is running.
It should not write back into the repository or assume git is present.

Characteristics:

- Runs inside Docker or another deployed environment.
- Writes runtime state to `EVAL_STATE_DIR`.
- Writes eval outputs to `EVAL_OUTPUT_DIR`.
- UI edits to ground truth are stored as runtime overlay annotations.
- Runtime overlays can later be exported and manually promoted into repo data.

## Data Flow

```text
checked-in corpus + checked-in annotations
        +
runtime annotation overlay
        |
        v
effective dataset
        |
        v
eval engine CLI
        |
        +--> model A resultset
        +--> model B resultset
        +--> run metadata
        |
        v
scoring layer
        |
        v
UI / recommendation
```

The engine always runs against an effective dataset. In development, the
effective dataset may be written under `data/`. In deployed mode, it is built
from repo baseline data plus runtime overlay data and written under the state
directory.

## Inputs

### Effective Dataset

The effective dataset represents the issue corpus and current ground-truth
certification state.

Required fields per issue:

```json
{
  "issue_number": 1190,
  "title": "AppSpec JSON Schema",
  "body": "Issue body text...",
  "state": "open",
  "html_url": "https://github.com/digitalocean/doctl/issues/1190",
  "ground_truth": {
    "status": "certified",
    "label": "documentation",
    "method": "manual_adjudication",
    "confidence": "high",
    "rationale": "The issue asks for publishing a JSON Schema in documentation."
  }
}
```

Ground truth status values:

- `certified`: safe to include in scored metrics.
- `uncertified`: classify in unscored analysis only.
- `needs_adjudication`: conflicting or low-confidence evidence; classify but do
  not score unless explicitly included by run config.

### Model Catalog

The model catalog defines provider model IDs and pricing. Prices must be
traceable so cost can be recomputed from token usage.

```json
{
  "models": {
    "llama-8b": {
      "display_name": "Llama 3.1 8B Instruct",
      "provider": "digitalocean_si",
      "provider_model": "llama-3.1-8b-instruct",
      "input_price_per_1m_tokens": 0.05,
      "output_price_per_1m_tokens": 0.10,
      "pricing_source": "config/model_catalog.json"
    }
  }
}
```

### Run Config

Run config is passed by CLI flags and environment variables.

```json
{
  "model_ids": ["llama-8b", "llama-70b"],
  "concurrency": 8,
  "timeout_seconds": 30,
  "max_retries": 2,
  "temperature": 0,
  "max_output_tokens": 64,
  "streaming": false,
  "dataset_path": "data/eval_dataset.json",
  "output_dir": "runs"
}
```

Concurrency must be visible in persisted run metadata and UI metrics because
latency and throughput depend on it.

## Prompt Contract

Each issue is classified independently. Multiple issues must not be batched into
one prompt.

The model should receive issue title and body, but not maintainer labels used for
ground truth certification. Hiding maintainer labels prevents the model from
copying the evaluation target.

Expected model output:

```json
{
  "label": "bug",
  "rationale": "The issue describes a command behavior defect."
}
```

Allowed labels:

- `bug`
- `enhancement`
- `question`
- `documentation`
- `security`
- `other`

## Per-Call Result Contract

Each issue-model call produces one durable result record.

```json
{
  "call_id": "run_001:llama-8b:1190",
  "run_id": "run_001",
  "model_id": "llama-8b",
  "issue_number": 1190,
  "status": "ok",
  "attempts": 1,
  "retryable": false,
  "request": {
    "temperature": 0,
    "max_output_tokens": 64,
    "timeout_seconds": 30,
    "streaming": false
  },
  "response": {
    "raw_output": "{\"label\":\"documentation\",\"rationale\":\"...\"}",
    "parsed_label": "documentation",
    "rationale": "The issue asks for a published schema in documentation.",
    "parse_error": null
  },
  "usage": {
    "input_tokens": 420,
    "output_tokens": 31,
    "total_tokens": 451,
    "usage_source": "provider"
  },
  "cost": {
    "input_price_per_1m_tokens": 0.05,
    "output_price_per_1m_tokens": 0.10,
    "input_cost_usd": 0.000021,
    "output_cost_usd": 0.0000031,
    "total_cost_usd": 0.0000241,
    "pricing_source": "config/model_catalog.json"
  },
  "timing": {
    "queued_at": "2026-06-08T20:00:00.000Z",
    "started_at": "2026-06-08T20:00:00.015Z",
    "request_sent_at": "2026-06-08T20:00:00.018Z",
    "first_token_at": null,
    "response_completed_at": "2026-06-08T20:00:00.842Z",
    "ended_at": "2026-06-08T20:00:00.844Z",
    "queue_wait_ms": 15,
    "time_to_first_token_ms": null,
    "generation_ms": null,
    "total_latency_ms": 824,
    "wall_time_ms": 844,
    "measurement_source": "client_non_streaming"
  },
  "error": null
}
```

For non-streaming runs, `time_to_first_token_ms` and `generation_ms` are `null`.
For streaming runs, `first_token_at`, TTFT, and generation duration should be
captured when the provider supports streaming.

## Per-Call Metrics

Per-call metrics are recorded for every issue-model call. These fields are the
source of truth for run-level p50/p95 latency, throughput, cost, and error-rate
summaries.

### Timing Metrics

- `queue_wait_ms`: Time from the task entering the engine queue to the task
  acquiring a concurrency slot. This measures local engine backpressure, not
  provider latency.
- `total_latency_ms`: Client-observed request latency from `request_sent_at` to
  `response_completed_at`. This is the primary latency metric used for p50 and
  p95 reporting.
- `wall_time_ms`: Time from `queued_at` to `ended_at`, including queue wait,
  request latency, parsing, retry bookkeeping, and local overhead.
- `time_to_first_token_ms`: Time from `request_sent_at` to `first_token_at`.
  This is only available for streaming runs and is `null` for non-streaming
  runs.
- `generation_ms`: Time from `first_token_at` to `response_completed_at`. This
  is only available for streaming runs and is `null` for non-streaming runs.

The UI should display the concurrency used for the run beside latency metrics,
because p50/p95 values are only meaningful with that context.

### Cost Metrics

- `input_tokens`: Prompt tokens reported by the provider, or estimated only if
  the run explicitly enables token estimation.
- `output_tokens`: Completion tokens reported by the provider, or estimated only
  if the run explicitly enables token estimation.
- `input_cost_usd`: `input_tokens / 1_000_000 *
  input_price_per_1m_tokens`.
- `output_cost_usd`: `output_tokens / 1_000_000 *
  output_price_per_1m_tokens`.
- `total_cost_usd`: `input_cost_usd + output_cost_usd`.
- `usage_source`: `provider`, `estimated`, or `missing`.
- `pricing_source`: Model catalog source used for the token rates.

If provider usage is missing and estimation is not enabled, token counts and
cost fields should be `null` rather than silently guessed.

### Output Quality Fields

- `raw_output`: Exact model output for inspection and parsing audits.
- `parsed_label`: Parsed label if the model returned a valid target class.
- `rationale`: Parsed rationale when available.
- `parse_error`: Parsing failure details if the response could not be converted
  into the expected JSON contract.

### Failure Metrics

- `status`: `ok` or `error`.
- `attempts`: Number of attempts made for this issue-model call.
- `retryable`: Whether this failed call can be retried independently.
- `error.type`: One of the fixed error types below.
- `error.http_status`: HTTP status when applicable.

Failed calls remain in the resultset and contribute to operational summaries.
They are not dropped from the denominator.

### Error Result

Failures are persisted as first-class results so they can be retried
individually and counted in operational metrics.

```json
{
  "call_id": "run_001:llama-8b:1190",
  "run_id": "run_001",
  "model_id": "llama-8b",
  "issue_number": 1190,
  "status": "error",
  "attempts": 3,
  "retryable": true,
  "response": {
    "raw_output": null,
    "parsed_label": null,
    "rationale": null,
    "parse_error": null
  },
  "usage": null,
  "cost": {
    "total_cost_usd": 0
  },
  "timing": {
    "total_latency_ms": 30000,
    "wall_time_ms": 30016,
    "measurement_source": "client_non_streaming"
  },
  "error": {
    "type": "timeout",
    "message": "Request timed out after 30 seconds",
    "http_status": null
  }
}
```

Error type enum:

- `rate_limit`
- `timeout`
- `api_error`
- `auth_error`
- `client_error`
- `parse_error`
- `invalid_label`
- `network_error`
- `other`

## Run Metadata Contract

Each run persists metadata that makes the resultsets reproducible and
interpretable.

```json
{
  "run_id": "run_001",
  "created_at": "2026-06-08T20:00:00Z",
  "completed_at": "2026-06-08T20:03:12Z",
  "wall_clock_ms": 192000,
  "dataset_id": "doctl_issues_530",
  "dataset_path": "data/eval_dataset.json",
  "model_ids": ["llama-8b", "llama-70b"],
  "prompt_source": "config/prompts/classification_template.txt",
  "model_catalog_path": "config/model_catalog.json",
  "concurrency": 8,
  "timeout_seconds": 30,
  "max_retries": 2,
  "temperature": 0,
  "max_output_tokens": 64,
  "streaming": false,
  "label_schema": [
    "bug",
    "enhancement",
    "question",
    "documentation",
    "security",
    "other"
  ]
}
```

## Persistence Layout

### Checked-In Repository Artifacts

Repository data contains stable inputs and curated artifacts.

```text
data/
  doctl_issues.json
  ground_truth_annotations.json
  eval_dataset.json
  baseline_runs/

config/
  model_catalog.json
  prompts/
    classification_template.txt

scripts/
  run_eval.py
  score_results.py

src/eval_harness/
```

`baseline_runs/` is optional. If checked in, it should contain selected,
representative persisted results used for review, not every local or deployed
run.

### Runtime State

Runtime artifacts are not committed. They are written under configurable
directories.

Environment variables:

- `EVAL_STATE_DIR`: annotation overlays, UI edits, audit log, effective dataset.
- `EVAL_OUTPUT_DIR`: eval run outputs.

Default local paths:

```text
state/
runs/
```

Default container paths:

```text
/app/state
/app/runs
```

Runtime layout:

```text
state/
  annotations/
    manual_adjudications.json
    audit_log.jsonl
  effective_dataset.json

runs/
  {run_id}/
    run.json
    results/
      {model_id}.json
    summaries/
      operational_metrics.json
      scored_metrics.json
      unscored_analysis.json
```

The UI should write label edits to runtime annotation overlays, not directly to
checked-in data files. Promotion from runtime state to repo data is a separate
development workflow.

## Code Organization

Initial implementation should keep the CLI small while separating reusable
engine components.

```text
src/eval_harness/
  __init__.py
  dataset.py        # Load corpus, annotations, overlays, effective dataset.
  models.py         # Load model catalog and resolve provider model config.
  prompt.py         # Build classification prompts from the selected template.
  client.py         # OpenAI-compatible API client.
  runner.py         # Async execution, concurrency, retries.
  resultset.py      # Result schemas, serialization, run IDs.
  timing.py         # Timing capture and latency metric helpers.
  cost.py           # Token usage and cost calculation.
  errors.py         # Error classification and retryability.
  scoring.py        # Metrics over persisted resultsets.

scripts/
  run_eval.py       # CLI entrypoint for eval engine.
  score_results.py  # CLI entrypoint for recomputing summaries.
```

The engine should persist raw resultsets before scoring. This allows scoring
logic to change without rerunning expensive inference.

## CLI Shape

Example local run:

```bash
uv run python scripts/run_eval.py \
  --dataset data/eval_dataset.json \
  --prompt config/prompts/classification_template.txt \
  --models llama-8b,llama-70b \
  --concurrency 8 \
  --output-dir runs
```

Example container run:

```bash
docker run \
  -e DIGITALOCEAN_SI_API_KEY=... \
  -e EVAL_STATE_DIR=/app/state \
  -e EVAL_OUTPUT_DIR=/app/runs \
  -v "$(pwd)/runtime-state:/app/state" \
  -v "$(pwd)/runs:/app/runs" \
  glowing-carnival
```

Retry failed calls from a previous resultset:

```bash
uv run python scripts/run_eval.py \
  --retry-failed runs/run_001/results/llama-8b.json \
  --output-dir runs
```

## Design Principles

- One issue-model pair produces one persisted call result.
- The model does not receive maintainer labels used for evaluation truth.
- Raw model output is always saved.
- Failed calls are saved and counted, not dropped.
- Cost is calculated from token usage and model catalog pricing.
- Concurrency is part of the run config and persisted metadata.
- Runtime annotation edits are overlays, not direct modifications to repo data.
- Scoring and UI read persisted results instead of duplicating inference logic.
