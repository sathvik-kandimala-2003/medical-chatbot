"""Interactive Plotly figures built from InteractionStore data.

Every function takes a DataFrame already produced by
InteractionStore.fetch_all_df() / fetch_chunk_traces_df() and returns a
plotly Figure - no data fetching or aggregation happens in here, so these
stay easy to reuse (dashboard page, notebooks, tests) and easy to test in
isolation.
"""

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

EMPTY_MESSAGE = "Not enough data yet - ask the chatbot a few questions first."


def _empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        annotations=[{"text": EMPTY_MESSAGE, "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 14}}],
    )
    return fig


def latency_over_time(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_figure("Response Latency Over Time")
    fig = px.line(
        df, x="timestamp", y="total_response_time_ms", markers=True,
        title="Response Latency Over Time",
        labels={"timestamp": "Time", "total_response_time_ms": "Total Response Time (ms)"},
    )
    return fig


def retrieval_score_distribution(chunks_df: pd.DataFrame) -> go.Figure:
    """FAISS and BM25 score distributions overlaid (different scales/directions -
    FAISS is L2 distance, lower = closer; BM25 is a term-weight score, higher = closer)."""
    if chunks_df.empty:
        return _empty_figure("Retrieval Score Distribution")

    fig = go.Figure()
    faiss_scores = chunks_df["faiss_score"].dropna()
    bm25_scores = chunks_df["bm25_score"].dropna()
    if not faiss_scores.empty:
        fig.add_trace(go.Histogram(x=faiss_scores, name="FAISS (L2 distance, lower=closer)", opacity=0.65))
    if not bm25_scores.empty:
        fig.add_trace(go.Histogram(x=bm25_scores, name="BM25 (higher=closer)", opacity=0.65))
    fig.update_layout(
        title="Retrieval Score Distribution", barmode="overlay",
        xaxis_title="Score", yaxis_title="Chunk Count",
    )
    return fig


def reranker_score_distribution(chunks_df: pd.DataFrame) -> go.Figure:
    if chunks_df.empty:
        return _empty_figure("Reranker Score Distribution")
    scores = chunks_df["rerank_score"].dropna()
    if scores.empty:
        return _empty_figure("Reranker Score Distribution")
    fig = px.histogram(
        x=scores, nbins=30, title="Reranker (Cross-Encoder) Score Distribution",
        labels={"x": "Cross-Encoder Score (higher=more relevant)"},
    )
    fig.update_layout(yaxis_title="Chunk Count", showlegend=False)
    return fig


def token_usage_over_time(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_figure("Token Usage Over Time")
    melted = df.melt(
        id_vars=["timestamp"],
        value_vars=["estimated_prompt_tokens", "estimated_output_tokens"],
        var_name="token_type", value_name="tokens",
    )
    melted["token_type"] = melted["token_type"].map({
        "estimated_prompt_tokens": "Prompt Tokens", "estimated_output_tokens": "Completion Tokens",
    })
    fig = px.line(
        melted, x="timestamp", y="tokens", color="token_type", markers=True,
        title="Token Usage Over Time", labels={"timestamp": "Time", "tokens": "Tokens"},
    )
    return fig


def context_size_over_time(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_figure("Final Context Size Over Time")
    fig = px.line(
        df, x="timestamp", y="final_context_chars", markers=True,
        title="Final Context Size Over Time",
        labels={"timestamp": "Time", "final_context_chars": "Context Size (chars)"},
    )
    return fig


def top_referenced_documents(chunks_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    if chunks_df.empty:
        return _empty_figure("Top Referenced Documents")
    counts = chunks_df["source"].value_counts().head(top_n).reset_index()
    counts.columns = ["source", "times_retrieved"]
    fig = px.bar(
        counts, x="times_retrieved", y="source", orientation="h",
        title="Top Referenced Documents", labels={"times_retrieved": "Times Retrieved", "source": "Document"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig


def most_frequently_retrieved_pages(chunks_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    if chunks_df.empty:
        return _empty_figure("Most Frequently Retrieved Pages")
    working = chunks_df.copy()
    working["source_page"] = working["source"].astype(str) + " (p. " + working["page"].astype(str) + ")"
    counts = working["source_page"].value_counts().head(top_n).reset_index()
    counts.columns = ["source_page", "times_retrieved"]
    fig = px.bar(
        counts, x="times_retrieved", y="source_page", orientation="h",
        title="Most Frequently Retrieved Pages",
        labels={"times_retrieved": "Times Retrieved", "source_page": "Document Page"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig


def response_time_histogram(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_figure("Query Response Time Distribution")
    fig = px.histogram(
        df, x="total_response_time_ms", nbins=30, title="Query Response Time Distribution",
        labels={"total_response_time_ms": "Total Response Time (ms)"},
    )
    fig.update_layout(yaxis_title="Query Count")
    return fig


def query_timeline(df: pd.DataFrame) -> go.Figure:
    """Each query as a point in time, colored by latency and labeled on hover -
    a quick visual read of usage volume/bursts plus which queries were slow."""
    if df.empty:
        return _empty_figure("Query Timeline")
    fig = px.scatter(
        df, x="timestamp", y="total_response_time_ms", color="total_response_time_ms",
        hover_data=["user_query"], color_continuous_scale="Bluered",
        title="Query Timeline", labels={"timestamp": "Time", "total_response_time_ms": "Response Time (ms)"},
    )
    return fig
