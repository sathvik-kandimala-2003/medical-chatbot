"""Runs a QA dataset through the chatbot's existing hybrid retrieval pipeline
(and, optionally, answer generation) and scores the results.

Reuses retrieval.HybridRetriever.retrieve_with_trace() and
gemini_service.generate_gemini_answer_detailed() directly - this module
computes evaluation metrics on top of them, it does not reimplement
retrieval or generation.

A note on Recall@K and Coverage, since both need a notion of "how many
relevant items exist in total" that a loosely-labeled QA dataset can't fully
answer: this evaluator doesn't have exhaustive relevance judgments over the
whole corpus (that would mean hand-labeling every chunk against every
question). Recall@K instead treats the number of distinct ground-truth
signals named for a question (expected_sources / relevant_chunks entries) as
a lower-bound estimate of "total relevant" - see _estimate_total_relevant().
This is a standard, documented simplification for hand-authored evaluation
sets; it will under-count recall's denominator if a question's true relevant
set is broader than what was labeled, so treat Recall@K here as an upper
bound on true recall, not an exact measurement.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Set

from evaluation.datasets.schema import QAItem
from evaluation.metrics import (
    average_precision,
    coverage,
    f1_score,
    hit,
    is_relevant,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from gemini_service import GeminiGenerationError, generate_gemini_answer_detailed
from retrieval import ChunkTrace, HybridRetriever

logger = logging.getLogger(__name__)


@dataclass
class QuestionEvalResult:
    question: str
    ground_truth_answer: Optional[str]
    retrieved_sources: List[str]
    chunk_traces: List[ChunkTrace]
    relevance_flags: List[bool]
    total_relevant_estimate: int
    precision_at_k: float
    recall_at_k: float
    f1: float
    reciprocal_rank: float
    average_precision: float
    hit: bool
    ndcg_at_k: float
    retrieval_time_ms: float
    generation_time_ms: float
    generated_answer: Optional[str] = None
    generation_error: Optional[str] = None


@dataclass
class EvaluationReport:
    k: int
    generated_at: str
    per_question: List[QuestionEvalResult] = field(default_factory=list)
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    f1_score: float = 0.0
    mrr: float = 0.0
    map_score: float = 0.0
    hit_rate: float = 0.0
    ndcg_at_k: float = 0.0
    coverage: float = 0.0
    avg_retrieval_time_ms: float = 0.0
    avg_generation_time_ms: float = 0.0

    def as_display_rows(self):
        return [
            ("Precision@K", f"{self.precision_at_k:.3f}"),
            ("Recall@K", f"{self.recall_at_k:.3f}"),
            ("F1 Score", f"{self.f1_score:.3f}"),
            ("MRR", f"{self.mrr:.3f}"),
            ("MAP", f"{self.map_score:.3f}"),
            ("Hit Rate", f"{self.hit_rate:.3f}"),
            ("nDCG@K", f"{self.ndcg_at_k:.3f}"),
            ("Coverage", f"{self.coverage:.3f}"),
            ("Average Retrieval Time", f"{self.avg_retrieval_time_ms:.1f} ms"),
            ("Average Generation Time", f"{self.avg_generation_time_ms:.1f} ms"),
        ]


class Evaluator:
    def __init__(self, hybrid_retriever: HybridRetriever, k: int = 5, generate_answers: bool = True):
        self._retriever = hybrid_retriever
        self._k = k
        self._generate_answers = generate_answers

    def evaluate_dataset(self, dataset: List[QAItem]) -> EvaluationReport:
        logger.info(
            "Evaluating %d question(s) at k=%d (generate_answers=%s)",
            len(dataset), self._k, self._generate_answers,
        )
        results = [self._evaluate_one(item) for item in dataset]

        report = EvaluationReport(
            k=self._k, generated_at=datetime.now(timezone.utc).isoformat(), per_question=results,
        )
        self._aggregate(report)
        return report

    def _evaluate_one(self, item: QAItem) -> QuestionEvalResult:
        start = time.perf_counter()
        docs, _metrics, chunk_traces = self._retriever.retrieve_with_trace(item.question, top_n=self._k)
        retrieval_time_ms = (time.perf_counter() - start) * 1000

        relevance_flags = [is_relevant(doc, item) for doc in docs]
        total_relevant = _estimate_total_relevant(item, relevance_flags)

        precision = precision_at_k(relevance_flags, self._k)
        recall = recall_at_k(relevance_flags, self._k, total_relevant)

        generated_answer, generation_time_ms, generation_error = None, 0.0, None
        if self._generate_answers and docs:
            gen_start = time.perf_counter()
            try:
                answer = generate_gemini_answer_detailed(item.question, docs)
                generated_answer = answer.text
            except GeminiGenerationError as exc:
                generation_error = str(exc)
                logger.warning("Generation failed for question %r: %s", item.question, exc)
            generation_time_ms = (time.perf_counter() - gen_start) * 1000

        return QuestionEvalResult(
            question=item.question,
            ground_truth_answer=item.ground_truth_answer,
            # basename, to match BM25Index.unique_sources() - see coverage() in _aggregate()
            retrieved_sources=[os.path.basename(doc.metadata.get("source") or "") for doc in docs],
            chunk_traces=chunk_traces,
            relevance_flags=relevance_flags,
            total_relevant_estimate=total_relevant,
            precision_at_k=precision,
            recall_at_k=recall,
            f1=f1_score(precision, recall),
            reciprocal_rank=reciprocal_rank(relevance_flags),
            average_precision=average_precision(relevance_flags, total_relevant),
            hit=hit(relevance_flags),
            ndcg_at_k=ndcg_at_k(relevance_flags, self._k),
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms,
            generated_answer=generated_answer,
            generation_error=generation_error,
        )

    def _aggregate(self, report: EvaluationReport) -> None:
        results = report.per_question
        n = len(results)
        if n == 0:
            return

        report.precision_at_k = sum(r.precision_at_k for r in results) / n
        report.recall_at_k = sum(r.recall_at_k for r in results) / n
        report.f1_score = sum(r.f1 for r in results) / n
        report.mrr = sum(r.reciprocal_rank for r in results) / n
        report.map_score = sum(r.average_precision for r in results) / n
        report.hit_rate = sum(1.0 for r in results if r.hit) / n
        report.ndcg_at_k = sum(r.ndcg_at_k for r in results) / n
        report.avg_retrieval_time_ms = sum(r.retrieval_time_ms for r in results) / n
        report.avg_generation_time_ms = sum(r.generation_time_ms for r in results) / n

        retrieved_sources = {s for r in results for s in r.retrieved_sources if s}
        corpus_sources = self._corpus_sources()
        report.coverage = coverage(retrieved_sources, corpus_sources) if corpus_sources else 0.0

    def _corpus_sources(self) -> Set[str]:
        bm25_index = self._retriever.bm25_index
        if bm25_index is None:
            return set()
        return bm25_index.unique_sources()


def _estimate_total_relevant(item: QAItem, retrieved_relevance_flags: List[bool]) -> int:
    """See module docstring: total-relevant isn't exhaustively known, so this
    uses the number of distinct ground-truth signals as a lower-bound proxy,
    widened to at least however many were actually found relevant in top-k
    (recall can't exceed 1.0 by definition)."""
    ground_truth_signal_count = max(
        len(item.expected_sources), len(item.relevant_chunks), 1 if item.relevant_documents else 0,
    )
    retrieved_relevant_count = sum(retrieved_relevance_flags)
    return max(ground_truth_signal_count, retrieved_relevant_count)
