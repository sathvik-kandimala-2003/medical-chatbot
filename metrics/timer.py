"""Lightweight stage-timing helper used across the retrieval/generation pipeline."""

import time
from contextlib import contextmanager


class Stopwatch:
    """Measures the wall-clock duration of a `with` block, in milliseconds.

    Usage:
        sw = Stopwatch()
        with sw.measure():
            do_work()
        sw.elapsed_ms
    """

    def __init__(self):
        self.elapsed_ms: float = 0.0

    @contextmanager
    def measure(self):
        start = time.perf_counter()
        try:
            yield self
        finally:
            self.elapsed_ms = (time.perf_counter() - start) * 1000
