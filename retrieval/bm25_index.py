"""BM25 keyword search over the same chunks indexed into FAISS.

The index is built once at ingestion time (see create_memory_for_llm.py) and
persisted to disk so the app doesn't need to re-tokenize the whole corpus on
every startup.
"""

import logging
import os
import pickle
import re
from pathlib import Path
from typing import List, Set, Tuple, Union

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Lowercase, punctuation-stripped word tokenizer for BM25 scoring."""
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    """Wraps rank_bm25.BM25Okapi with the Document objects it was built from,
    so search() can return Documents rather than bare indices/scores.
    """

    def __init__(self, documents: List[Document]):
        self._documents = documents
        corpus_tokens = [_tokenize(doc.page_content) for doc in documents]
        self._bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None

    def __len__(self) -> int:
        return len(self._documents)

    def unique_sources(self) -> Set[str]:
        """Every distinct source document basename indexed - used by the
        offline evaluator's Coverage metric (evaluation/metrics/retrieval_metrics.py)."""
        return {
            os.path.basename(doc.metadata.get("source") or "")
            for doc in self._documents
            if doc.metadata.get("source")
        }

    def search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Return up to k (Document, score) pairs, highest score first."""
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(_tokenize(query))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._documents[i], float(scores[i])) for i in top_indices]

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._documents, f)
        logger.info("Saved BM25 index (%d documents) to %s", len(self._documents), path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BM25Index":
        path = Path(path)
        with open(path, "rb") as f:
            documents = pickle.load(f)
        index = cls(documents)
        logger.info("Loaded BM25 index (%d documents) from %s", len(index), path)
        return index

    @classmethod
    def build(cls, documents: List[Document]) -> "BM25Index":
        logger.info("Building BM25 index from %d documents", len(documents))
        return cls(documents)
