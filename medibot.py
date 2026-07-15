import contextlib
import logging
import os

# Windows + HuggingFace cache fixes (must run before heavy imports)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_NO_TQDM", "1")
os.environ.setdefault("DISABLE_TQDM", "1")

import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from gemini_service import (
    GeminiGenerationError,
    NO_CONTEXT_MESSAGE,
    format_sources,
    generate_gemini_answer,
)

load_dotenv(find_dotenv())

logging.basicConfig(
    filename="medibot_error.log",
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DB_FAISS_PATH = "vectorstore/db_faiss"
RETRIEVAL_K = 4

LLM_UNAVAILABLE_MESSAGE = (
    "The AI assistant is temporarily unavailable. Please try again in a moment."
)


@st.cache_resource
def load_embeddings():
    try:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"local_files_only": True},
            )
    except Exception:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def load_vectorstore():
    embeddings = load_embeddings()
    return FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve_documents(vectorstore, query, k=RETRIEVAL_K):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)


def main():
    st.set_page_config(page_title="Ask Sathvik's Chatbot!", page_icon="💬", layout="wide")
    st.title("Ask Sathvik's Chatbot!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    prompt = st.chat_input("Pass your prompt here")
    if not prompt:
        return

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        with st.spinner("Searching medical records and generating an answer..."):
            vectorstore = load_vectorstore()
            docs = retrieve_documents(vectorstore, prompt)

            if not docs:
                result_to_show = NO_CONTEXT_MESSAGE
            else:
                try:
                    answer = generate_gemini_answer(prompt, docs)
                except GeminiGenerationError:
                    logger.exception("Gemini generation failed for query: %s", prompt)
                    st.error("The AI assistant hit an error while generating a response.")
                    result_to_show = LLM_UNAVAILABLE_MESSAGE
                else:
                    if answer.strip() == NO_CONTEXT_MESSAGE:
                        result_to_show = NO_CONTEXT_MESSAGE
                    else:
                        sources = format_sources(docs)
                        result_to_show = f"{answer}\n\n{sources}" if sources else answer

        st.chat_message("assistant").markdown(result_to_show)
        st.session_state.messages.append({"role": "assistant", "content": result_to_show})
    except Exception:
        logger.exception("Unexpected error handling query: %s", prompt)
        st.error("Something went wrong while processing your question.")
        result_to_show = LLM_UNAVAILABLE_MESSAGE
        st.chat_message("assistant").markdown(result_to_show)
        st.session_state.messages.append({"role": "assistant", "content": result_to_show})


if __name__ == "__main__":
    main()
