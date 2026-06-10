# Model Inventory Snapshot

## Purpose

This step discovers the DigitalOcean Serverless Inference models visible to the configured model access key. It normalizes model metadata and prepares explainable candidate lists for later smoke testing. It does not run issue classification or call chat completions.

## Snapshot Semantics

The inventory is a point-in-time snapshot. DigitalOcean model availability can change, and the visible model set can be scoped by the configured model access key.

## Scope

The empirical model pool focuses on DigitalOcean Serverless Inference models. Dedicated deployments, BYOM, and self-hosted GPU alternatives are handled qualitatively outside this inventory.

## Eligibility

A model is smoke-test eligible when it is visible through `/v1/models`, appears suitable for chat-style text generation, is not clearly an embedding/reranker/image/audio/video/TTS/async-only model, and is not explicitly excluded by metadata.

## Pricing

Pricing is supplied by `config/model_metadata.json`. Missing pricing is a metadata gap, not a smoke-test exclusion. Smoke testing can proceed without pricing, but pilot and final evaluation cost metrics require input and output token prices.

## Known Limitations

- `/v1/models` may not expose complete capability or context-window metadata.
- The snapshot only reflects models visible to the key at fetch time.
- Capability inference from model IDs is conservative and routes uncertain rows to review.
- Pricing must be manually maintained and source-attributed.

## Counts

- Total visible models: 68
- Smoke eligible models: 51
- Smoke eligible, pricing needed: 35
- Cost-ready eligible models: 16
- Pricing needed: 35
- Excluded models: 13
- Needs metadata review: 4
- Missing pricing: 52

## Artifacts

- `data/models/raw_models_response.json`: raw models response
- `data/models/models_snapshot.json`: models snapshot
- `data/models/models_inventory.csv`: models inventory
- `data/models/eligible_models.csv`: eligible models
- `data/models/cost_ready_models.csv`: cost ready models
- `data/models/pricing_needed.csv`: pricing needed
- `data/models/excluded_models.csv`: excluded models
- `data/models/needs_metadata_review.csv`: needs metadata review
- `data/models/model_inventory_summary.json`: model inventory summary
- `data/models/model_inventory.md`: model inventory markdown
