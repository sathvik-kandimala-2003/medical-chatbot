"""Data captured about individual chunks during a single retrieve_with_trace() call.

Kept separate from metrics.QueryMetrics (the aggregate per-query timing/count
numbers already used by the chatbot sidebar): ChunkTrace is per-chunk detail
consumed by the analytics dashboard and offline evaluator, not by the chat UI.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ChunkTrace:
    rank: int
    content_preview: str
    source: str
    page: Optional[str]

    # FAISS uses L2 distance by default (lower = more similar) - unlike
    # BM25/cross-encoder scores, where higher = more relevant. None means the
    # chunk wasn't surfaced by that retriever (e.g. BM25-only match).
    faiss_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
