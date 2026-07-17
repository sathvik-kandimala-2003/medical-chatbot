"""SQLite-backed persistence for chatbot interactions.

One flat table: the QueryMetrics fields are stored as real columns (so
session analytics can use plain SQL/pandas aggregation - AVG, MIN, MAX -
without re-deriving anything), and the per-chunk ChunkTrace list is stored as
a JSON blob column (nested/variable-length, not a good fit for flat SQL
columns; the volume here - dozens of chunks per query, not millions of rows -
doesn't justify a second joined table).

A short-lived connection is opened per operation rather than held open for
the app's lifetime, since Streamlit may run script reruns on different
threads and sqlite3 connections aren't safe to share across threads.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import pandas as pd

from metrics import QueryMetrics
from retrieval import ChunkTrace

from .models import InteractionRecord

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "dashboard/storage/data/interactions.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    session_id TEXT,
    user_query TEXT NOT NULL,
    standalone_query TEXT,
    answer TEXT,
    embedding_time_ms REAL,
    vector_search_time_ms REAL,
    bm25_search_time_ms REAL,
    fusion_time_ms REAL,
    reranking_time_ms REAL,
    llm_generation_time_ms REAL,
    total_response_time_ms REAL,
    retrieved_documents_count INTEGER,
    final_context_chunk_count INTEGER,
    final_context_chars INTEGER,
    estimated_prompt_tokens INTEGER,
    estimated_output_tokens INTEGER,
    estimated_cost_usd REAL,
    chunks_json TEXT
);
"""

_METRICS_COLUMNS = [
    "embedding_time_ms", "vector_search_time_ms", "bm25_search_time_ms",
    "fusion_time_ms", "reranking_time_ms", "llm_generation_time_ms",
    "total_response_time_ms", "retrieved_documents_count",
    "final_context_chunk_count", "final_context_chars",
    "estimated_prompt_tokens", "estimated_output_tokens",
]


class InteractionStore:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def log_interaction(self, record: InteractionRecord) -> int:
        chunks_json = json.dumps([asdict(chunk) for chunk in record.chunks])
        metrics_values = [getattr(record.metrics, col) for col in _METRICS_COLUMNS]

        with self._connect() as conn:
            cursor = conn.execute(
                f"""
                INSERT INTO interactions (
                    timestamp, session_id, user_query, standalone_query, answer,
                    {", ".join(_METRICS_COLUMNS)},
                    estimated_cost_usd, chunks_json
                ) VALUES ({", ".join(["?"] * (5 + len(_METRICS_COLUMNS) + 2))})
                """,
                [
                    record.timestamp, record.session_id, record.user_query,
                    record.standalone_query, record.answer,
                    *metrics_values,
                    record.estimated_cost_usd, chunks_json,
                ],
            )
            interaction_id = cursor.lastrowid

        logger.debug("Logged interaction %s (query=%r)", interaction_id, record.user_query)
        return interaction_id

    def fetch_all_df(self) -> pd.DataFrame:
        """One row per interaction. chunks_json stays a raw JSON string column;
        use fetch_chunk_traces_df() for a flattened per-chunk view."""
        with self._connect() as conn:
            df = pd.read_sql_query("SELECT * FROM interactions ORDER BY id", conn)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def fetch_chunk_traces_df(self) -> pd.DataFrame:
        """One row per (interaction, retrieved chunk) - for retrieval-analytics
        views and score-distribution charts."""
        interactions = self.fetch_all_df()
        if interactions.empty:
            return pd.DataFrame(columns=[
                "interaction_id", "timestamp", "user_query", "rank", "content_preview",
                "source", "page", "faiss_score", "bm25_score", "rrf_score", "rerank_score",
            ])

        rows = []
        for _, interaction in interactions.iterrows():
            for chunk in json.loads(interaction["chunks_json"] or "[]"):
                rows.append({
                    "interaction_id": interaction["id"],
                    "timestamp": interaction["timestamp"],
                    "user_query": interaction["user_query"],
                    **chunk,
                })
        return pd.DataFrame(rows)

    def count(self) -> int:
        with self._connect() as conn:
            (n,) = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()
        return n

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM interactions")
        logger.info("Cleared all interaction history from %s", self.db_path)
