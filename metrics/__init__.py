"""Framework-agnostic runtime metrics for the retrieval + generation pipeline.

Public API:
    QueryMetrics    - one query's timings/counts (see models.py)
    Stopwatch       - `with Stopwatch().measure():` timing helper (see timer.py)
    estimate_tokens - char-based token estimate fallback (see token_estimator.py)
"""

from .models import QueryMetrics
from .timer import Stopwatch
from .token_estimator import estimate_tokens

__all__ = ["QueryMetrics", "Stopwatch", "estimate_tokens"]
