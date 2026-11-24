"""Lightweight USD pricing table per 1M tokens for common LLM models."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    input_per_1m: float
    output_per_1m: float
    cached_input_per_1m: float = 0.0


# Prices in USD per 1,000,000 tokens (users can override or extend at runtime)
DEFAULT_PRICING: dict[str, ModelPricing] = {
    "claude-sonnet-4": ModelPricing(3.00, 15.00, 0.30),
    "claude-opus-4": ModelPricing(15.00, 75.00, 1.50),
    "claude-haiku-3.5": ModelPricing(0.80, 4.00, 0.08),
    "gpt-4o": ModelPricing(2.50, 10.00, 1.25),
    "gpt-4o-mini": ModelPricing(0.15, 0.60, 0.075),
    "gemini-2.5-pro": ModelPricing(1.25, 10.00, 0.31),
    "gemini-2.5-flash": ModelPricing(0.15, 0.60, 0.0375),
    "local": ModelPricing(0.0, 0.0, 0.0),
}


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    custom_pricing: dict[str, ModelPricing] | None = None,
) -> float:
    """Calculate estimated USD cost for a single LLM call."""
    table = {**DEFAULT_PRICING, **(custom_pricing or {})}
    pricing = table.get(model)

    if pricing is None:
        # Fallback to partial prefix matching (e.g., versioned model strings)
        for key, value in table.items():
            if model.startswith(key):
                pricing = value
                break

    if pricing is None:
        return 0.0

    billable_input = max(0, input_tokens - cached_tokens)
    cost = (
        (billable_input / 1_000_000) * pricing.input_per_1m
        + (cached_tokens / 1_000_000) * pricing.cached_input_per_1m
        + (output_tokens / 1_000_000) * pricing.output_per_1m
    )
    return round(cost, 8)
