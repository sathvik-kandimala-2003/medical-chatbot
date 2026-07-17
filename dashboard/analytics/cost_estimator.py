"""Estimated USD cost of a Gemini call, from the token counts gemini_service
already reports (real usage_metadata when available, heuristic estimate
otherwise - see gemini_service._extract_token_counts).

Pricing is a rough, editable estimate, not a billing source of truth: Gemini
pricing varies by tier/region and changes over time, and the app's model
(GEMINI_MODEL_NAME in gemini_service.py) can be swapped independently of this
file. Update the two rates below to match your actual plan for accurate
numbers; treat the dashboard's "Estimated Total API Cost" as directional.
"""

from dataclasses import dataclass

# USD per 1M tokens - defaults approximate the Gemini Flash-Lite free/low tier
# pricing at time of writing.
DEFAULT_INPUT_PRICE_PER_MILLION_TOKENS = 0.075
DEFAULT_OUTPUT_PRICE_PER_MILLION_TOKENS = 0.30


@dataclass(frozen=True)
class CostRates:
    input_price_per_million_tokens: float = DEFAULT_INPUT_PRICE_PER_MILLION_TOKENS
    output_price_per_million_tokens: float = DEFAULT_OUTPUT_PRICE_PER_MILLION_TOKENS


def estimate_cost_usd(prompt_tokens: int, output_tokens: int, rates: CostRates = CostRates()) -> float:
    input_cost = (prompt_tokens / 1_000_000) * rates.input_price_per_million_tokens
    output_cost = (output_tokens / 1_000_000) * rates.output_price_per_million_tokens
    return input_cost + output_cost
