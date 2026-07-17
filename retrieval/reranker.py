"""Cross-encoder reranking of fused retrieval candidates.

A cross-encoder scores (query, chunk) pairs jointly, which is far more
accurate than the cosine/BM25 similarity used for initial retrieval - but
too slow to run over the whole corpus, which is why it only sees the
already-fused top candidates.
"""

import contextlib
import logging
import os
from typing import List, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from .config import CROSS_ENCODER_MODEL_NAME

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str = CROSS_ENCODER_MODEL_NAME):
        self.model_name = model_name
        self._model = self._load_model(model_name)

    @staticmethod
    def _load_model(model_name: str) -> CrossEncoder:
        # Mirrors medibot.load_embeddings(): try a silent offline load first
        # (avoids noisy HF symlink warnings on Windows once weights are
        # cached), then fall back to a normal load that downloads if needed.
        try:
            with open(os.devnull, "w") as devnull, \
                    contextlib.redirect_stdout(devnull), \
                    contextlib.redirect_stderr(devnull):
                return CrossEncoder(model_name, local_files_only=True)
        except Exception:
            logger.info("Local weights for %s not found; downloading", model_name)
            return CrossEncoder(model_name)

    def rerank_with_scores(
        self, query: str, documents: List[Document], top_n: int = 5
    ) -> List[Tuple[Document, float]]:
        """Score each document against the query and return the top_n (doc, score), best first."""
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._model.predict(pairs)

        ranked = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)
        top_ranked = [(doc, float(score)) for doc, score in ranked[:top_n]]

        logger.debug(
            "Cross-encoder reranked %d candidates -> top %d",
            len(documents), len(top_ranked),
        )
        return top_ranked

    def rerank(self, query: str, documents: List[Document], top_n: int = 5) -> List[Document]:
        """Same as rerank_with_scores(), but returns just the documents."""
        return [doc for doc, _score in self.rerank_with_scores(query, documents, top_n=top_n)]
