"""Hybrid retrieval pipeline: FAISS + BM25 -> Reciprocal Rank Fusion -> CrossEncoder rerank.

Public API:
    HybridRetriever   - orchestrates the full pipeline (see hybrid_retriever.py)
    BM25Index         - keyword search index, buildable/persistable (see bm25_index.py)
    CrossEncoderReranker - reranks fused candidates (see reranker.py)
    reciprocal_rank_fusion - fuses ranked lists (see fusion.py)
    RetrievalConfig   - tunable pipeline parameters (see config.py)
    ChunkTrace        - per-chunk score/rank detail from retrieve_with_trace() (see models.py)

    See loaders.py for factory functions that assemble a HybridRetriever from disk
    (used by medibot.py, connect_memory_with_llm.py, and the offline evaluator).
"""

from .bm25_index import BM25Index
from .config import RetrievalConfig
from .fusion import reciprocal_rank_fusion
from .hybrid_retriever import HybridRetriever
from .models import ChunkTrace
from .reranker import CrossEncoderReranker

__all__ = [
    "BM25Index",
    "ChunkTrace",
    "CrossEncoderReranker",
    "HybridRetriever",
    "RetrievalConfig",
    "reciprocal_rank_fusion",
]
