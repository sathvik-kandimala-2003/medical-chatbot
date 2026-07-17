"""Orchestrates the full hybrid retrieval pipeline:

    query -> FAISS top-k  ---\
                               +--> Reciprocal Rank Fusion --> CrossEncoder rerank --> top-N
    query -> BM25 top-k   ---/

BM25 index and reranker are optional. If either is missing or fails at
runtime, retrieval degrades to a smaller pipeline instead of raising, so an
existing deployment (e.g. a vectorstore built before this upgrade, with no
bm25_index.pkl yet) keeps working exactly as before.

Every call runs through _run_pipeline(), a single retrieval pass that
collects a QueryMetrics (embedding timed separately from the FAISS search
itself, via embeddings.embed_query() + a *_by_vector() call, rather than the
bundled vectorstore.as_retriever().invoke()) and, at no extra retrieval cost,
the per-chunk FAISS/BM25/RRF/rerank scores behind each stage. retrieve() and
retrieve_with_metrics() keep their original signatures/behavior exactly;
retrieve_with_trace() is the only way to also get the per-chunk scores
(used by the analytics dashboard and offline evaluator, not by the chat UI).
"""

import logging
import os
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from metrics import QueryMetrics, Stopwatch

from .bm25_index import BM25Index
from .config import RetrievalConfig
from .fusion import _doc_key, reciprocal_rank_fusion_with_scores
from .models import ChunkTrace
from .reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(
        self,
        vectorstore,
        embeddings,
        bm25_index: Optional[BM25Index] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        config: Optional[RetrievalConfig] = None,
    ):
        self._vectorstore = vectorstore
        self._embeddings = embeddings
        self._bm25_index = bm25_index
        self._reranker = reranker
        self._config = config or RetrievalConfig()

        if bm25_index is None:
            logger.warning("HybridRetriever initialized without a BM25 index; retrieval will be FAISS-only.")
        if reranker is None:
            logger.warning("HybridRetriever initialized without a cross-encoder reranker; rerank stage will be skipped.")

    @property
    def bm25_index(self) -> Optional[BM25Index]:
        """Read-only access for callers that need corpus-level info (e.g. the
        offline evaluator's Coverage metric via bm25_index.unique_sources())."""
        return self._bm25_index

    def retrieve(self, query: str, top_n: Optional[int] = None) -> List[Document]:
        """Backward-compatible: run the pipeline and return only the documents."""
        docs, _metrics, _traces = self._run_pipeline(query, top_n)
        return docs

    def retrieve_with_metrics(
        self, query: str, top_n: Optional[int] = None
    ) -> Tuple[List[Document], QueryMetrics]:
        """Run FAISS + BM25 -> RRF -> cross-encoder rerank, returning the
        top_n Documents plus a QueryMetrics with per-stage timings/counts.

        Note: QueryMetrics.total_response_time_ms and the token/generation
        fields are left at 0 here - this method only covers retrieval.
        Callers that also run LLM generation (see medibot.py) fill those in
        to get a true end-to-end total.
        """
        docs, metrics, _traces = self._run_pipeline(query, top_n)
        return docs, metrics

    def retrieve_with_trace(
        self, query: str, top_n: Optional[int] = None
    ) -> Tuple[List[Document], QueryMetrics, List[ChunkTrace]]:
        """Same as retrieve_with_metrics(), plus a ChunkTrace per returned
        document carrying its FAISS/BM25/RRF/rerank scores and final rank."""
        return self._run_pipeline(query, top_n)

    def _run_pipeline(
        self, query: str, top_n: Optional[int]
    ) -> Tuple[List[Document], QueryMetrics, List[ChunkTrace]]:
        top_n = top_n if top_n is not None else self._config.rerank_top_n
        metrics = QueryMetrics()

        faiss_scored = self._search_faiss(query, metrics)
        faiss_docs = [doc for doc, _score in faiss_scored]
        faiss_score_by_key = {_doc_key(doc): score for doc, score in faiss_scored}

        ranked_lists = [faiss_docs]
        bm25_scored = self._search_bm25(query, metrics)
        bm25_score_by_key = {}
        if bm25_scored is not None:
            ranked_lists.append([doc for doc, _score in bm25_scored])
            bm25_score_by_key = {_doc_key(doc): score for doc, score in bm25_scored}

        fusion_sw = Stopwatch()
        with fusion_sw.measure():
            fused_scored = reciprocal_rank_fusion_with_scores(
                ranked_lists, k=self._config.rrf_k, top_n=self._config.fusion_top_n
            )
        metrics.fusion_time_ms = fusion_sw.elapsed_ms
        fused_docs = [doc for doc, _score in fused_scored]
        rrf_score_by_key = {_doc_key(doc): score for doc, score in fused_scored}
        metrics.retrieved_documents_count = len(fused_docs)
        logger.debug("RRF fusion produced %d candidate document(s)", len(fused_docs))

        reranked_scored = self._rerank(query, fused_docs, top_n, metrics)
        results = [doc for doc, _score in reranked_scored]
        rerank_score_by_key = {
            _doc_key(doc): score for doc, score in reranked_scored if score is not None
        }

        metrics.final_context_chunk_count = len(results)
        metrics.final_context_chars = sum(len(doc.page_content) for doc in results)

        chunk_traces = [
            _build_chunk_trace(
                rank, doc, faiss_score_by_key, bm25_score_by_key, rrf_score_by_key, rerank_score_by_key
            )
            for rank, doc in enumerate(results, start=1)
        ]

        logger.info(
            "Hybrid retrieval returned %d document(s) for query %r "
            "(embed=%.1fms, vector_search=%.1fms, bm25=%.1fms, fusion=%.1fms, rerank=%.1fms)",
            len(results), query,
            metrics.embedding_time_ms, metrics.vector_search_time_ms,
            metrics.bm25_search_time_ms, metrics.fusion_time_ms, metrics.reranking_time_ms,
        )
        return results, metrics, chunk_traces

    def _search_faiss(self, query: str, metrics: QueryMetrics) -> List[Tuple[Document, float]]:
        embed_sw = Stopwatch()
        with embed_sw.measure():
            query_vector = self._embeddings.embed_query(query)
        metrics.embedding_time_ms = embed_sw.elapsed_ms

        search_sw = Stopwatch()
        with search_sw.measure():
            # L2 distance by default (lower = more similar) - see ChunkTrace docstring.
            scored = self._vectorstore.similarity_search_with_score_by_vector(
                query_vector, k=self._config.faiss_k
            )
        metrics.vector_search_time_ms = search_sw.elapsed_ms

        # numpy float32 -> native float, so scores serialize cleanly to JSON/SQLite downstream.
        scored = [(doc, float(score)) for doc, score in scored]

        logger.debug("FAISS search returned %d document(s)", len(scored))
        return scored

    def _search_bm25(self, query: str, metrics: QueryMetrics) -> Optional[List[Tuple[Document, float]]]:
        if self._bm25_index is None:
            return None

        bm25_sw = Stopwatch()
        try:
            with bm25_sw.measure():
                scored = self._bm25_index.search(query, self._config.bm25_k)
            metrics.bm25_search_time_ms = bm25_sw.elapsed_ms
            logger.debug("BM25 search returned %d document(s)", len(scored))
            return scored
        except Exception:
            metrics.bm25_search_time_ms = bm25_sw.elapsed_ms
            logger.exception("BM25 search failed; continuing with FAISS results only")
            return None

    def _rerank(
        self, query: str, fused_docs: List[Document], top_n: int, metrics: QueryMetrics
    ) -> List[Tuple[Document, Optional[float]]]:
        if self._reranker is None:
            return [(doc, None) for doc in fused_docs[:top_n]]

        rerank_sw = Stopwatch()
        try:
            with rerank_sw.measure():
                results = self._reranker.rerank_with_scores(query, fused_docs, top_n=top_n)
            metrics.reranking_time_ms = rerank_sw.elapsed_ms
            return results
        except Exception:
            metrics.reranking_time_ms = rerank_sw.elapsed_ms
            logger.exception("Cross-encoder reranking failed; falling back to fused RRF order")
            return [(doc, None) for doc in fused_docs[:top_n]]


def _build_chunk_trace(rank, doc, faiss_scores, bm25_scores, rrf_scores, rerank_scores) -> ChunkTrace:
    key = _doc_key(doc)
    metadata = doc.metadata or {}
    page = metadata.get("page")
    page_label = str(page) if isinstance(page, int) else metadata.get("page_label")
    return ChunkTrace(
        rank=rank,
        content_preview=doc.page_content[:200],
        source=os.path.basename(metadata.get("source") or "Unknown document"),
        page=page_label,
        faiss_score=faiss_scores.get(key),
        bm25_score=bm25_scores.get(key),
        rrf_score=rrf_scores.get(key),
        rerank_score=rerank_scores.get(key),
    )
