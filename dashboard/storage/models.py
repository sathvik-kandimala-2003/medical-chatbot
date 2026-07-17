"""One record per chatbot turn, built entirely from data the chatbot's
existing pipeline already produced - metrics.QueryMetrics (runtime timings/
counts) and retrieval.ChunkTrace (per-chunk retrieval scores). This module
adds no new measurement, only a container for persisting what already exists.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from metrics import QueryMetrics
from retrieval import ChunkTrace


@dataclass
class InteractionRecord:
    timestamp: str  # ISO 8601, UTC
    session_id: str
    user_query: str
    standalone_query: str
    answer: str
    metrics: QueryMetrics
    chunks: List[ChunkTrace] = field(default_factory=list)
    estimated_cost_usd: float = 0.0
    id: Optional[int] = None
