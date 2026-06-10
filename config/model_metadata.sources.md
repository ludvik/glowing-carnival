# Model Metadata Sources

Prices were copied from the DigitalOcean Inference Pricing docs.

- Pricing captured date: 2026-06-10
- Units: USD per 1M input tokens and USD per 1M output tokens
- Standard prices are used; off-peak pricing is intentionally ignored for apples-to-apples comparison.
- Only the curated smoke-pool models are enriched initially.
- Additional visible models can be priced later if they survive into pilot or final evaluation.

The `/v1/models` endpoint is used for visibility, context window, and max output token metadata when available. It does not provide pricing, so pricing is maintained in `config/model_metadata.json`.
