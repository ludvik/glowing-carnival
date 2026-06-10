## Evaluation Methodology and Key Decisions

This project treats the exercise as a production model-evaluation problem, not just a leaderboard. The customer scenario is that a high-volume issue-classification workflow may be overpaying for a frontier model. My goal is therefore to determine which model should actually run in production, with evidence around quality, cost, latency, throughput, and failure handling.

### Corpus and Ground Truth

The full corpus contains 530 GitHub issues from the DigitalOcean `doctl` repository. Every issue is included in the model classification corpus and is classified by each selected model. Only a high-confidence subset is used for scored quality metrics.

I do not treat GitHub maintainer labels as authoritative ground truth. They are useful weak signals, but they were applied by different people over multiple years and do not map one-to-one to the customer’s six-label schema:

- `bug`
- `enhancement`
- `question`
- `documentation`
- `security`
- `other`

The scored dataset is built in two stages:

1. A deterministic labeling script maps high-signal maintainer labels and text heuristics into candidate labels.
2. A manual adjudication pass corrects obvious schema mismatches and improves thin or risky classes such as `documentation`, `question`, and `security`.

The current dataset split is:

| Split | Count |
|---|---:|
| Scored | 131 |
| Unscored | 395 |
| Review | 4 |
| Total | 530 |

The scored-label distribution is:

| Label | Support |
|---|---:|
| `bug` | 35 |
| `enhancement` | 35 |
| `security` | 26 |
| `question` | 18 |
| `documentation` | 15 |
| `other` | 2 |

The `other` class is intentionally conservative. I do not map unlabeled or ambiguous issues to `other`; I only use it for issues that genuinely do not fit the first five classes. Because `other` has very low scored support, I report its metrics for transparency but do not use it as a decisive model-selection signal.

The `security` class has more support, but most high-confidence security examples in this corpus are dependency vulnerability or CVE-style issues. I therefore treat security metrics as directional rather than a complete measure of all possible security-sensitive support issues.

### Model Inventory and Candidate Pool

Model selection is also treated as a funnel. I first capture a point-in-time inventory of models visible to my DigitalOcean Serverless Inference model access key through `/v1/models`. This snapshot found:

| Stage | Count |
|---|---:|
| Visible models | 68 |
| Smoke-eligible text/chat models | 51 |
| Curated, fully priced smoke-pool models | 16 |
| Broader smoke-eligible models still missing pricing | 35 |
| Excluded non-text / non-final candidates | 13 |
| Needs metadata review | 4 |

I keep the broader model universe visible, but I do not attempt to price or run every model up front. Instead, I curate a 16-model smoke pool that covers meaningful production tradeoffs:

- cost-efficient / small-ish models
- balanced mid-tier models
- larger quality-oriented models
- commercial or frontier-reference models
- reasoning-style versus non-reasoning-style behavior

This avoids starting with only two hand-picked models while also avoiding a brute-force evaluation of every visible model. The smoke pool is fully enriched with source-attributed pricing metadata. Pricing is maintained as a local snapshot because `/v1/models` exposes availability and model metadata, but not token pricing.

Routers are excluded from the empirical comparison because they are not single-model candidates. They may be useful in a production architecture, but they make per-model cost, latency, and failure attribution less clean. Dedicated Inference, BYOM, and self-hosted GPU deployments are also treated as production-scale alternatives rather than part of the empirical Serverless Inference comparison.

I did not run all 51 models immediately. This is not intended to be an exhaustive benchmark; it is a customer-facing model selection exercise. Many of the remaining models are near-duplicates within a provider family, lack pricing metadata, or do not represent a materially different production tradeoff. Running all of them would increase cost and noise without necessarily improving the recommendation. 

Instead, I curated a 16-model smoke pool that spans the tradeoff space the customer cares about: cost-efficient models, balanced mid-tier models, larger quality-oriented models, commercial reference models, and one reasoning-style candidate. The goal of the smoke pool is to identify which models are practically viable: callable, reliable, able to follow the strict six-label output contract, and reasonable on latency and cost.

The final recommended models should come from models that pass this staged funnel. If I discover a gap during smoke or pilot evaluation, I can add a model in a documented second version of the pool, but I would not silently change the candidate set after seeing results.

### Evaluation Funnel

The evaluation flow is:

```text
stable doctl issue corpus
  -> high-confidence scored subset + full unscored corpus
  -> DigitalOcean Serverless Inference model inventory
  -> curated 16-model smoke pool
  -> smoke test for callability, output contract, latency, and errors
  -> pilot evaluation on a stratified scored subset
  -> finalist evaluation on the full corpus
  -> production recommendation

  I treated model selection as a funnel rather than starting with two hand-picked models. First, I captured the point-in-time model universe visible to my DigitalOcean Serverless Inference key through `/v1/models`. That produced 68 visible models. After excluding non-text modalities, routers, and models that were not appropriate for direct chat-style classification, I had 51 smoke-eligible text/chat candidates.

