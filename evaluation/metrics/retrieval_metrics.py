"""Pure retrieval-quality metric functions.

Each per-question metric takes `relevance`: a list of booleans (was the
chunk at this rank relevant?), already produced by relevance.is_relevant()
for one question's retrieved chunks, in rank order. MRR and MAP are the
*mean* of reciprocal_rank()/average_precision() across all questions in the
dataset - that averaging happens in the evaluator, not in here, since it
needs the whole dataset's results at once.
"""

import math
from typing import Sequence, Set


def precision_at_k(relevance: Sequence[bool], k: int) -> float:
    top_k = relevance[:k]
    if not top_k:
        return 0.0
    return sum(top_k) / len(top_k)


def recall_at_k(relevance: Sequence[bool], k: int, total_relevant: int) -> float:
    if total_relevant == 0:
        return 0.0
    return sum(relevance[:k]) / total_relevant


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def reciprocal_rank(relevance: Sequence[bool]) -> float:
    """1 / rank of the first relevant hit; 0.0 if none were relevant."""
    for rank, relevant in enumerate(relevance, start=1):
        if relevant:
            return 1.0 / rank
    return 0.0


def average_precision(relevance: Sequence[bool], total_relevant: int) -> float:
    """Mean of precision@k evaluated at each rank where a relevant chunk was hit."""
    if total_relevant == 0:
        return 0.0
    hits = 0
    precision_at_hits = []
    for rank, relevant in enumerate(relevance, start=1):
        if relevant:
            hits += 1
            precision_at_hits.append(hits / rank)
    if not precision_at_hits:
        return 0.0
    return sum(precision_at_hits) / total_relevant


def hit(relevance: Sequence[bool]) -> bool:
    """Was at least one relevant chunk retrieved at all?"""
    return any(relevance)


def ndcg_at_k(relevance: Sequence[bool], k: int) -> float:
    """Binary-relevance normalized DCG@k (relevant=1, not relevant=0)."""
    top_k = relevance[:k]
    dcg = sum(1.0 / math.log2(rank + 1) for rank, relevant in enumerate(top_k, start=1) if relevant)
    ideal_hit_count = min(sum(relevance), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hit_count + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def coverage(retrieved_sources: Set[str], corpus_sources: Set[str]) -> float:
    """Catalog coverage: fraction of the corpus's source documents that were
    surfaced at least once across the whole evaluation run. A diversity
    signal (is retrieval only ever pulling from a handful of documents?),
    distinct from any single query's precision/recall.
    """
    if not corpus_sources:
        return 0.0
    return len(retrieved_sources & corpus_sources) / len(corpus_sources)
