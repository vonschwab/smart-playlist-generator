from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenPricing:
    input_per_million: float
    output_per_million: float


MODEL_PRICING_USD: dict[str, TokenPricing] = {
    "gpt-4o-mini": TokenPricing(input_per_million=0.15, output_per_million=0.60),
    "gpt-4.1-mini": TokenPricing(input_per_million=0.40, output_per_million=1.60),
    "gpt-4.1-nano": TokenPricing(input_per_million=0.10, output_per_million=0.40),
    "gpt-5-nano": TokenPricing(input_per_million=0.05, output_per_million=0.40),
    "gpt-5-mini": TokenPricing(input_per_million=0.25, output_per_million=2.00),
    "gpt-5.4-nano": TokenPricing(input_per_million=0.20, output_per_million=1.25),
    "gpt-5.4-mini": TokenPricing(input_per_million=0.75, output_per_million=4.50),
}


def estimate_cost_usd(model: str, *, input_tokens: int, output_tokens: int) -> float | None:
    pricing = MODEL_PRICING_USD.get(model)
    if pricing is None:
        return None
    return (input_tokens / 1_000_000 * pricing.input_per_million) + (
        output_tokens / 1_000_000 * pricing.output_per_million
    )
