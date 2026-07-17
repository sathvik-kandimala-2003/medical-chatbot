"""Heuristic token estimation, used only when an exact count isn't available.

~4 characters per token is a standard rule of thumb for English text across
GPT/Gemini-style tokenizers. gemini_service.generate_gemini_answer_detailed()
prefers Gemini's own reported usage_metadata token counts when present and
falls back to this estimator only if that's missing.
"""


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, round(len(text) / 4))
