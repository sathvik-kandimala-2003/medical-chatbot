import logging
import os
import uuid
from datetime import datetime, timezone

# Windows + HuggingFace cache fixes (must run before heavy imports)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_NO_TQDM", "1")
os.environ.setdefault("DISABLE_TQDM", "1")

import streamlit as st
from dotenv import load_dotenv, find_dotenv

from dashboard.analytics import estimate_cost_usd
from dashboard.storage import InteractionRecord, InteractionStore
from gemini_service import (
    GeminiGenerationError,
    NO_CONTEXT_MESSAGE,
    condense_question,
    format_sources,
    generate_gemini_answer_detailed,
)
from memory import ConversationMemory
from metrics import Stopwatch
from retrieval import HybridRetriever, RetrievalConfig
from retrieval.loaders import (
    load_bm25_index_safe,
    load_embeddings_model,
    load_faiss_vectorstore,
    load_reranker_safe,
)

load_dotenv(find_dotenv())

# LOG_LEVEL lets ops flip on retrieval-pipeline INFO/DEBUG logs (stage
# timings, fallbacks) without editing code; defaults to the original
# errors-only behavior so nothing changes for existing deployments.
logging.basicConfig(
    filename="medibot_error.log",
    level=os.environ.get("LOG_LEVEL", "ERROR").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

DB_FAISS_PATH = "vectorstore/db_faiss"
RETRIEVAL_K = 4
RETRIEVAL_CONFIG = RetrievalConfig()

LLM_UNAVAILABLE_MESSAGE = (
    "The AI assistant is temporarily unavailable. Please try again in a moment."
)



# Thin @st.cache_resource wrappers around retrieval/loaders.py's plain
# factory functions, so the actual loading logic lives in exactly one place
# shared with connect_memory_with_llm.py and the offline evaluator.

@st.cache_resource
def load_embeddings():
    return load_embeddings_model()


@st.cache_resource
def load_vectorstore():
    return load_faiss_vectorstore(load_embeddings(), db_path=DB_FAISS_PATH)


def retrieve_documents(vectorstore, query, k=RETRIEVAL_K):
    """Plain FAISS top-k retrieval. Kept as-is for backward compatibility;
    the app itself now retrieves via HybridRetriever (see load_hybrid_retriever)."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)


@st.cache_resource
def load_bm25_index():
    return load_bm25_index_safe()


@st.cache_resource
def load_reranker():
    return load_reranker_safe()


@st.cache_resource
def load_hybrid_retriever():
    return HybridRetriever(
        vectorstore=load_vectorstore(),
        embeddings=load_embeddings(),
        bm25_index=load_bm25_index(),
        reranker=load_reranker(),
        config=RETRIEVAL_CONFIG,
    )


@st.cache_resource
def load_interaction_store() -> InteractionStore:
    """Shared across every session (it's just a SQLite file on disk) - this
    is what backs the Evaluation Dashboard page."""
    return InteractionStore()


def get_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


def log_interaction(session_id, user_query, standalone_query, answer, metrics, chunks):
    """Best-effort: persisting analytics data must never break the chat."""
    try:
        cost = estimate_cost_usd(metrics.estimated_prompt_tokens, metrics.estimated_output_tokens)
        record = InteractionRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=session_id,
            user_query=user_query,
            standalone_query=standalone_query,
            answer=answer,
            metrics=metrics,
            chunks=chunks,
            estimated_cost_usd=cost,
        )
        load_interaction_store().log_interaction(record)
    except Exception:
        logger.exception("Failed to log interaction for the analytics dashboard")


def get_conversation_memory() -> ConversationMemory:
    """Per-session chat memory.

    Must live in st.session_state, not st.cache_resource: cache_resource is
    shared by every user of the deployed app, so caching a ConversationMemory
    there would leak one user's chat history into another's.
    """
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = ConversationMemory()
    return st.session_state.conversation_memory


def render_metrics_sidebar():
    """Collapsible runtime-metrics panel, confined to the sidebar so the main
    chat UI stays unchanged."""
    with st.sidebar:
        with st.expander("Runtime Metrics", expanded=False):
            metrics = st.session_state.get("last_metrics")
            if metrics is None:
                st.caption("Ask a question to see performance metrics.")
                return
            rows = metrics.as_display_rows()
            table_md = "| Metric | Value |\n|---|---|\n" + "\n".join(
                f"| {label} | {value} |" for label, value in rows
            )
            st.markdown(table_md)


def main():
    st.set_page_config(page_title="Ask Sathvik's Chatbot!", page_icon="💬", layout="wide")
    st.title("Ask Sathvik's Chatbot!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    prompt = st.chat_input("Pass your prompt here")
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        conversation_memory = get_conversation_memory()
        history_text = conversation_memory.as_text()
        session_id = get_session_id()

        try:
            with st.spinner("Searching medical records and generating an answer..."):
                turn_timer = Stopwatch()
                with turn_timer.measure():
                    # Rewrite follow-ups like "What are its symptoms?" into a
                    # standalone query so retrieval sees the actual topic
                    # instead of a bare pronoun. No-op (and no extra API
                    # call) on the first turn, since there's no history yet.
                    standalone_query = condense_question(prompt, history_text)
                    if standalone_query != prompt:
                        logger.info("Condensed follow-up question %r -> %r", prompt, standalone_query)

                    hybrid_retriever = load_hybrid_retriever()
                    # retrieve_with_trace() = retrieve_with_metrics() + a
                    # per-chunk score/rank breakdown for the analytics
                    # dashboard; the metrics themselves are identical either way.
                    docs, metrics, chunk_traces = hybrid_retriever.retrieve_with_trace(standalone_query)

                    if not docs:
                        result_to_show = NO_CONTEXT_MESSAGE
                    else:
                        try:
                            gen_timer = Stopwatch()
                            with gen_timer.measure():
                                gemini_answer = generate_gemini_answer_detailed(
                                    prompt, docs, chat_history_text=history_text
                                )
                        except GeminiGenerationError:
                            logger.exception("Gemini generation failed for query: %s", prompt)
                            st.error("The AI assistant hit an error while generating a response.")
                            result_to_show = LLM_UNAVAILABLE_MESSAGE
                        else:
                            metrics.llm_generation_time_ms = gen_timer.elapsed_ms
                            metrics.estimated_prompt_tokens = gemini_answer.prompt_tokens
                            metrics.estimated_output_tokens = gemini_answer.output_tokens

                            if gemini_answer.text.strip() == NO_CONTEXT_MESSAGE:
                                result_to_show = NO_CONTEXT_MESSAGE
                            else:
                                sources = format_sources(docs)
                                result_to_show = (
                                    f"{gemini_answer.text}\n\n{sources}" if sources else gemini_answer.text
                                )

                    # Don't remember turns where generation hard-failed - an
                    # error message isn't useful conversational context.
                    if result_to_show != LLM_UNAVAILABLE_MESSAGE:
                        conversation_memory.add_user_message(prompt)
                        conversation_memory.add_ai_message(result_to_show)

                metrics.total_response_time_ms = turn_timer.elapsed_ms
                st.session_state.last_metrics = metrics

                log_interaction(session_id, prompt, standalone_query, result_to_show, metrics, chunk_traces)

            st.chat_message("assistant").markdown(result_to_show)
            st.session_state.messages.append({"role": "assistant", "content": result_to_show})
        except Exception:
            logger.exception("Unexpected error handling query: %s", prompt)
            st.error("Something went wrong while processing your question.")
            result_to_show = LLM_UNAVAILABLE_MESSAGE
            st.chat_message("assistant").markdown(result_to_show)
            st.session_state.messages.append({"role": "assistant", "content": result_to_show})

    render_metrics_sidebar()


if __name__ == "__main__":
    main()
