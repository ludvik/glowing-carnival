# Model Screening

Model screening is the first empirical screening step for the curated DigitalOcean
Serverless Inference model pool. It is not final model evaluation.

The screening run gates basic production viability:

- the model can be called through DigitalOcean SI chat completions
- the model follows the JSON output contract closely enough to parse
- the parsed label is one of the six allowed issue labels
- usage/token metadata is present for cost instrumentation
- error behavior is acceptable
- latency shape is reasonable enough to continue

The screening wrapper uses an output budget large enough to avoid truncating
thinking-model responses. Hidden reasoning token usage is treated as part of the
model's operational profile and should show up in latency, output-token volume,
and cost rather than as an artificial truncation failure.

The screening issue set is intentionally small and stratified. It includes a few
certified examples from each target label and selected boundary cases, but it is
too small for final quality ranking.

Models that pass model screening proceed to pilot evaluation on a larger scored
subset. Final recommendations should be based on the broader evaluation funnel:
scored-set quality metrics, operational reliability, latency percentiles, and
traceable cost.

Screening accuracy is only a sanity check. A model should not be selected or rejected
solely because of one or two screening-set classification mistakes if callability,
output contract adherence, and reliability are otherwise acceptable.
