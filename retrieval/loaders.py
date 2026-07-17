"""Plain (non-Streamlit-cached) factory functions for building the retrieval stack.

Three different entrypoints need to construct the same FAISS + BM25 +
CrossEncoder + HybridRetriever stack: the Streamlit app (medibot.py, which
wraps these in @st.cache_resource), the CLI companion script
(connect_memory_with_llm.py), and the offline evaluator (evaluation/). This
module is the single place that logic lives, so all three stay in sync
instead of re-implementing "load embeddings, load FAISS, load BM25 with a
fallback, load the reranker with a fallback" three times over.
"""

import contextlib
import logging
import os
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from .bm25_index import BM25Index
from .config import BM25_INDEX_PATH, RetrievalConfig
from .hybrid_retriever import HybridRetriever
from .reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DB_FAISS_PATH = "vectorstore/db_faiss"


def load_embeddings_model(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    # Try a silent offline load first (avoids noisy HF symlink warnings on
    # Windows once weights are cached), then fall back to a normal load that
    # downloads if needed.
    try:
        with open(os.devnull, "w") as devnull, \
                contextlib.redirect_stdout(devnull), \
                contextlib.redirect_stderr(devnull):
            return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"local_files_only": True})
    except Exception:
        return HuggingFaceEmbeddings(model_name=model_name)


def load_faiss_vectorstore(embeddings, db_path: str = DB_FAISS_PATH) -> FAISS:
    return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)


def load_bm25_index_safe(path: str = BM25_INDEX_PATH) -> Optional[BM25Index]:
    if not os.path.exists(path):
        logger.warning(
            "BM25 index not found at %s (run create_memory_for_llm.py to build it). "
            "Falling back to FAISS-only retrieval.",
            path,
        )
        return None
    try:
        return BM25Index.load(path)
    except Exception:
        logger.exception("Failed to load BM25 index; falling back to FAISS-only retrieval.")
        return None


def load_reranker_safe() -> Optional[CrossEncoderReranker]:
    try:
        return CrossEncoderReranker()
    except Exception:
        logger.exception("Failed to load cross-encoder reranker; skipping the rerank stage.")
        return None


def build_hybrid_retriever(config: Optional[RetrievalConfig] = None) -> HybridRetriever:
    """Load every component from disk and assemble a ready-to-use HybridRetriever.

    No caching here by design - callers that run inside Streamlit should wrap
    the individual load_* functions in @st.cache_resource themselves (see
    medibot.py); callers outside Streamlit (CLI scripts, the evaluator) can
    call this directly.
    """
    embeddings = load_embeddings_model()
    vectorstore = load_faiss_vectorstore(embeddings)
    return HybridRetriever(
        vectorstore=vectorstore,
        embeddings=embeddings,
        bm25_index=load_bm25_index_safe(),
        reranker=load_reranker_safe(),
        config=config or RetrievalConfig(),
    )
