"""Session Analytics: aggregate stats over stored interactions.

Every number here is a plain SUM/AVG/MIN/MAX over columns the chatbot's
existing metrics.QueryMetrics already populated per turn (see
dashboard/storage) - nothing is recomputed or re-measured.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class SessionSummary:
    total_queries: int = 0
    avg_response_time_ms: float = 0.0
    fastest_query_ms: float = 0.0
    slowest_query_ms: float = 0.0
    avg_retrieval_time_ms: float = 0.0
    avg_generation_time_ms: float = 0.0
    avg_prompt_tokens: float = 0.0
    avg_completion_tokens: float = 0.0
    avg_context_size_chars: float = 0.0
    avg_retrieved_chunks: float = 0.0
    estimated_total_cost_usd: float = 0.0

    def as_display_rows(self):
        return [
            ("Total Queries", str(self.total_queries)),
            ("Average Response Time", f"{self.avg_response_time_ms:.1f} ms"),
            ("Fastest Query", f"{self.fastest_query_ms:.1f} ms"),
            ("Slowest Query", f"{self.slowest_query_ms:.1f} ms"),
            ("Average Retrieval Time", f"{self.avg_retrieval_time_ms:.1f} ms"),
            ("Average Generation Time", f"{self.avg_generation_time_ms:.1f} ms"),
            ("Average Prompt Tokens", f"{self.avg_prompt_tokens:.0f}"),
            ("Average Completion Tokens", f"{self.avg_completion_tokens:.0f}"),
            ("Average Context Size", f"{self.avg_context_size_chars:.0f} chars"),
            ("Average Retrieved Chunks", f"{self.avg_retrieved_chunks:.1f}"),
            ("Estimated Total API Cost", f"${self.estimated_total_cost_usd:.4f}"),
        ]


def compute_session_summary(df: pd.DataFrame) -> SessionSummary:
    """df is the output of InteractionStore.fetch_all_df()."""
    if df.empty:
        return SessionSummary()

    # "Retrieval time" = every stage before generation (embedding through
    # reranking); QueryMetrics keeps these as five separate fields, so this
    # is a sum of already-measured numbers, not a new measurement.
    retrieval_time_ms = (
        df["embedding_time_ms"] + df["vector_search_time_ms"] + df["bm25_search_time_ms"]
        + df["fusion_time_ms"] + df["reranking_time_ms"]
    )

    return SessionSummary(
        total_queries=len(df),
        avg_response_time_ms=df["total_response_time_ms"].mean(),
        fastest_query_ms=df["total_response_time_ms"].min(),
        slowest_query_ms=df["total_response_time_ms"].max(),
        avg_retrieval_time_ms=retrieval_time_ms.mean(),
        avg_generation_time_ms=df["llm_generation_time_ms"].mean(),
        avg_prompt_tokens=df["estimated_prompt_tokens"].mean(),
        avg_completion_tokens=df["estimated_output_tokens"].mean(),
        avg_context_size_chars=df["final_context_chars"].mean(),
        avg_retrieved_chunks=df["final_context_chunk_count"].mean(),
        estimated_total_cost_usd=df["estimated_cost_usd"].sum(),
    )
