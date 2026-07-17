"""Evaluation Dashboard - a completely separate Streamlit page from the
chatbot (medibot.py). Streamlit auto-discovers files under pages/ next to
the app's entrypoint and adds them to the sidebar navigation, so nothing in
medibot.py's own UI needed to change for this to appear.

Everything here is read-only analytics over data medibot.py already logged
via dashboard.storage.InteractionStore - no retrieval, generation, or
metrics computation happens on this page.
"""

import streamlit as st

from dashboard.analytics import compute_session_summary
from dashboard.charts import plots
from dashboard.export import to_csv_bytes, to_json_bytes
from dashboard.storage import InteractionStore

st.set_page_config(page_title="Evaluation Dashboard", page_icon="📊", layout="wide")
st.title("📊 Evaluation Dashboard")
st.caption(
    "Analytics built entirely from interactions the chatbot already logged - "
    "nothing on this page recomputes retrieval, generation, or runtime metrics."
)


@st.cache_resource
def get_store() -> InteractionStore:
    return InteractionStore()


store = get_store()

if st.button("🔄 Refresh"):
    st.rerun()

df = store.fetch_all_df()
chunks_df = store.fetch_chunk_traces_df()

if df.empty:
    st.info("No chatbot interactions logged yet. Ask the chatbot a few questions on the main page, then come back here.")
    st.stop()

# ---------------------------------------------------------------------------
# Session Analytics
# ---------------------------------------------------------------------------
st.header("Session Analytics")
summary = compute_session_summary(df)
summary_rows = summary.as_display_rows()
cols = st.columns(4)
for i, (label, value) in enumerate(summary_rows):
    cols[i % 4].metric(label, value)

st.divider()

# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
st.header("Visualizations")
tab_latency, tab_retrieval, tab_docs = st.tabs(["Latency & Tokens", "Retrieval & Reranking", "Documents"])

with tab_latency:
    c1, c2 = st.columns(2)
    c1.plotly_chart(plots.latency_over_time(df), width='stretch')
    c2.plotly_chart(plots.response_time_histogram(df), width='stretch')
    c1.plotly_chart(plots.token_usage_over_time(df), width='stretch')
    c2.plotly_chart(plots.context_size_over_time(df), width='stretch')
    st.plotly_chart(plots.query_timeline(df), width='stretch')

with tab_retrieval:
    c1, c2 = st.columns(2)
    c1.plotly_chart(plots.retrieval_score_distribution(chunks_df), width='stretch')
    c2.plotly_chart(plots.reranker_score_distribution(chunks_df), width='stretch')

with tab_docs:
    c1, c2 = st.columns(2)
    c1.plotly_chart(plots.top_referenced_documents(chunks_df), width='stretch')
    c2.plotly_chart(plots.most_frequently_retrieved_pages(chunks_df), width='stretch')

st.divider()

# ---------------------------------------------------------------------------
# Retrieval Analytics (per-query chunk breakdown)
# ---------------------------------------------------------------------------
st.header("Retrieval Analytics")
st.caption("Per-chunk FAISS / BM25 / RRF / reranker scores for a single query.")

query_labels = [
    f"#{row.id} · {row.timestamp:%Y-%m-%d %H:%M} · {row.user_query[:70]}"
    for row in df.itertuples()
]
selected_label = st.selectbox("Select a query to inspect", options=query_labels, index=len(query_labels) - 1)
selected_id = int(selected_label.split("·")[0].strip().lstrip("#"))
selected_row = df[df["id"] == selected_id].iloc[0]

st.markdown(f"**Query:** {selected_row['user_query']}")
if selected_row["standalone_query"] and selected_row["standalone_query"] != selected_row["user_query"]:
    st.caption(f"Condensed to standalone query: {selected_row['standalone_query']}")
answer_preview = selected_row["answer"] or ""
st.markdown(f"**Answer:** {answer_preview[:500]}{'…' if len(answer_preview) > 500 else ''}")

query_chunks = chunks_df[chunks_df["interaction_id"] == selected_id].sort_values("rank")
st.dataframe(
    query_chunks[[
        "rank", "source", "page", "faiss_score", "bm25_score", "rrf_score", "rerank_score", "content_preview",
    ]],
    width='stretch', hide_index=True,
)

st.divider()

# ---------------------------------------------------------------------------
# History (filterable, sortable, exportable)
# ---------------------------------------------------------------------------
st.header("History")

with st.expander("Filters", expanded=False):
    search_text = st.text_input("Search query text")
    latency_bounds = (
        float(df["total_response_time_ms"].min()),
        float(df["total_response_time_ms"].max()),
    )
    if latency_bounds[0] == latency_bounds[1]:
        latency_bounds = (latency_bounds[0], latency_bounds[1] + 1.0)
    min_latency, max_latency = st.slider(
        "Response time range (ms)", latency_bounds[0], latency_bounds[1], latency_bounds,
    )

history_df = df.copy()
if search_text:
    history_df = history_df[history_df["user_query"].str.contains(search_text, case=False, na=False)]
history_df = history_df[
    history_df["total_response_time_ms"].between(min_latency, max_latency)
]
history_df["total_tokens"] = history_df["estimated_prompt_tokens"] + history_df["estimated_output_tokens"]

history_display = history_df[[
    "timestamp", "user_query", "retrieved_documents_count", "total_response_time_ms",
    "estimated_prompt_tokens", "estimated_output_tokens", "total_tokens", "estimated_cost_usd",
]].rename(columns={
    "retrieved_documents_count": "retrieved_documents",
    "total_response_time_ms": "latency_ms",
    "estimated_prompt_tokens": "prompt_tokens",
    "estimated_output_tokens": "completion_tokens",
})

st.caption(f"{len(history_display)} of {len(df)} interactions shown. Click a column header to sort.")
st.dataframe(history_display, width='stretch', hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
st.header("Export")
c1, c2 = st.columns(2)
c1.download_button(
    "Download full history as CSV", data=to_csv_bytes(df),
    file_name="chatbot_interactions.csv", mime="text/csv",
)
c2.download_button(
    "Download full history as JSON", data=to_json_bytes(df),
    file_name="chatbot_interactions.json", mime="application/json",
)
