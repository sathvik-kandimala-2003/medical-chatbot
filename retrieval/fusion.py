"""Reciprocal Rank Fusion (RRF) for combining multiple ranked document lists.

RRF combines rankings without needing the retrievers' scores to be on
comparable scales (FAISS returns similarity/distance, BM25 returns a
corpus-dependent term-frequency score) - it only looks at each document's
rank position in each list.
"""

import logging
from typing import Dict, List, Sequence, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _doc_key(doc: Document) -> Tuple[str, str, str]:
    """Identity key used to recognize the "same" chunk across retrievers.

    FAISS and BM25 are built from the same chunk set, so identical chunks
    carry identical metadata; falling back to page_content guards against
    chunks that are missing source/page metadata.
    """
    metadata = doc.metadata or {}
    source = str(metadata.get("source", ""))
    page = str(metadata.get("page", metadata.get("page_label", "")))
    return (source, page, doc.page_content)


def reciprocal_rank_fusion_with_scores(
    ranked_lists: Sequence[List[Document]],
    k: int = 60,
    top_n: int = 20,
) -> List[Tuple[Document, float]]:
    """Fuse ranked document lists into a single ranking, keeping the RRF score.

    score(doc) = sum over each list containing doc of 1 / (k + rank),
    where rank is the document's 1-based position in that list. Documents
    absent from a list simply contribute 0 for that list.

    Args:
        ranked_lists: one ranked list of Documents per retriever.
        k: RRF damping constant; higher values reduce the influence of rank 1.
        top_n: number of fused documents to return.
    """
    scores: Dict[Tuple[str, str, str], float] = {}
    doc_lookup: Dict[Tuple[str, str, str], Document] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            doc_lookup.setdefault(key, doc)

    fused_keys = sorted(scores, key=lambda key: scores[key], reverse=True)[:top_n]
    fused = [(doc_lookup[key], scores[key]) for key in fused_keys]

    logger.debug(
        "RRF fused %d ranked list(s) (%d unique documents) into top %d",
        len(ranked_lists), len(scores), len(fused),
    )
    return fused


def reciprocal_rank_fusion(
    ranked_lists: Sequence[List[Document]],
    k: int = 60,
    top_n: int = 20,
) -> List[Document]:
    """Same as reciprocal_rank_fusion_with_scores(), but returns just the documents."""
    return [doc for doc, _score in reciprocal_rank_fusion_with_scores(ranked_lists, k=k, top_n=top_n)]
