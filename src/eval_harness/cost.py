from __future__ import annotations

from typing import Any


def calculate_cost(usage: dict[str, Any] | None, model: dict[str, Any]) -> dict[str, Any]:
    input_price = model["input_price_per_1m_tokens"]
    output_price = model["output_price_per_1m_tokens"]
    pricing_source = model.get("pricing_source", "config/model_catalog.json")

    if not usage:
        return {
            "input_price_per_1m_tokens": input_price,
            "output_price_per_1m_tokens": output_price,
            "input_cost_usd": None,
            "output_cost_usd": None,
            "total_cost_usd": None,
            "pricing_source": pricing_source,
        }

    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if input_tokens is None or output_tokens is None:
        return {
            "input_price_per_1m_tokens": input_price,
            "output_price_per_1m_tokens": output_price,
            "input_cost_usd": None,
            "output_cost_usd": None,
            "total_cost_usd": None,
            "pricing_source": pricing_source,
        }

    input_cost = input_tokens / 1_000_000 * input_price
    output_cost = output_tokens / 1_000_000 * output_price
    return {
        "input_price_per_1m_tokens": input_price,
        "output_price_per_1m_tokens": output_price,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": input_cost + output_cost,
        "pricing_source": pricing_source,
    }
