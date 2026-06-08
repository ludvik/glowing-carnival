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
