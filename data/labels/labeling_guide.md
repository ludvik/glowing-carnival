# Labeling Guide

This dataset is a high-confidence scored subset for the doctl issue
classification eval. It is not a complete relabeling of all issues.

## Target Labels

- `bug`: broken existing behavior, crashes, wrong results, ignored documented
  flags, invalid output, unexpected failures, or regressions.
- `enhancement`: new capability, new flag, API parity, UX improvement, platform
  support, output field, packaging, or workflow improvement.
- `question`: usage questions or troubleshooting requests without a clear
  product defect assertion.
- `documentation`: missing, outdated, incorrect, confusing, or inconsistent
  docs, README, tutorials, help text, usage text, examples, or installation
  instructions.
- `security`: CVEs, dependency vulnerabilities, credential exposure, token
  leakage, auth bypass, unsafe defaults, secret handling, or security-sensitive
  behavior.
- `other`: spam, tests, duplicate-only, off-topic, too little information, or
  impossible to classify. Unlabeled issues are not automatically `other`.

## Maintainer Labels Are Weak Signals

Maintainer labels are useful but noisy. The script maps only high-signal labels
to the target schema and treats workflow labels as context.

Primary weak-label mapping:

- `bug` -> `bug`
- `question` -> `question`
- `docs` -> `documentation`
- `security vulnerability` -> `security`
- `enhancement`, `suggestion`, `api-parity` -> `enhancement`

Secondary labels such as `packaging`, `snap`, `windows`, `wip`,
`waiting-response`, `good first issue`, `help wanted`, `Needs Investigation`,
`do-api`, `troubleshooting`, `blocked`, and `version 2.x` do not determine the
target label by themselves.

## Inclusion Criteria

An issue enters `scored_set` only when it has exactly one target label,
confidence is above the configured threshold, and no ambiguity/exclusion rule
applies. Manual scored overrides take precedence and require a rationale.

## Exclusion Criteria

Issues are excluded from scoring when they are unlabeled or low-confidence, have
conflicting primary labels, contain strong competing signals, have too little
information, or need human adjudication.

## Ambiguity Policy

Ambiguous multi-intent issues are routed to `review_queue.csv`. For example,
bug/question conflicts follow maintainer labels when present; otherwise they go
to review unless one signal clearly dominates. Documentation/enhancement
conflicts are scored only when the primary ask is clear.

## Limitation

This is a conservative, high-confidence scored subset. It is designed to support
defensible evaluation metrics, not to maximize coverage or replace human
labeling.

## Output Artifacts

- `classification_corpus.jsonl` contains every input issue exactly once with
  `split=scored`, `split=unscored`, or `split=review`.
- `review_queue.csv` contains automatically ambiguous cases that need human
  adjudication.
- `manual_review_candidates.csv` combines the review queue with high-value
  unscored candidates and low-confidence scored rows for optional manual review.
