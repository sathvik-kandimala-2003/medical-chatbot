"""Tunable parameters for the hybrid retrieval pipeline.

Centralizing these here means the numbers in the module docstrings/diagrams
(FAISS top-20, BM25 top-20, RRF fused top-20, rerank to top-5) live in one
place instead of being copy-pasted across modules.
"""

from dataclasses import dataclass

# Where the BM25 index is persisted, alongside the FAISS index directory.
BM25_INDEX_PATH = "vectorstore/bm25_index.pkl"

# cross-encoder/ms-marco-MiniLM-L-6-v2 is a small, fast reranker trained on
# MS MARCO passage ranking - a good accuracy/latency tradeoff for reranking
# ~20 candidates per query on CPU.
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class RetrievalConfig:
    """Candidate counts at each stage of the pipeline.

    faiss_k / bm25_k:   how many candidates each retriever contributes before fusion.
    rrf_k:              the RRF damping constant (score = 1 / (rrf_k + rank)); 60 is the
                         standard value from the original RRF paper and works well
                         without per-corpus tuning.
    fusion_top_n:       how many fused candidates survive to the reranking stage.
    rerank_top_n:       how many chunks are ultimately handed to Gemini.
    """

    faiss_k: int = 20
    bm25_k: int = 20
    rrf_k: int = 60
    fusion_top_n: int = 20
    rerank_top_n: int = 5
