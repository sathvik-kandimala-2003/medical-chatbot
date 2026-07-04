import contextlib
import os
import re
import traceback

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

load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"
LLM_MODEL_ID = "gpt2"
MAX_NEW_TOKENS = 128
RETRIEVAL_K = 3

CUSTOM_PROMPT_TEMPLATE = """
Use only the following context to answer the question.
If the answer is not contained in the context, say "I don't know."
Keep the response brief and factual.

Context:
{context}

Question:
{question}

Answer:
"""


def scrub_doc_text(text):
    text = re.sub(r"\s*Resources\b.*$", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\bReferences?\b.*$", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_prompt(question, docs):
    snippets = []
    for doc in docs[:RETRIEVAL_K]:
        text = scrub_doc_text(doc.page_content)
        if text:
            snippets.append(text)
    context = "\n\n".join(snippets)
    return CUSTOM_PROMPT_TEMPLATE.format(context=context, question=question)


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


def retrieve_documents(retriever, query):
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(query)
    if hasattr(retriever, "_get_relevant_documents"):
        return retriever._get_relevant_documents(query, run_manager=None)
    raise AttributeError("Retriever does not support get_relevant_documents")


def answer_from_docs(docs, question):
    snippets = []
    for doc in docs[:RETRIEVAL_K]:
        text = scrub_doc_text(doc.page_content)
        if text:
            snippets.append(text)
    if not snippets:
        return "Sorry, I couldn't find any relevant information in the documents."

    combined = " ".join(snippets)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", combined) if s.strip()]
    answer = []

    if re.search(r"\bdiabetes\b", question, re.IGNORECASE):
        for sentence in sentences:
            if re.search(r"\bdiabetes\b|\bglucose\b|\burination\b|\bthirst\b|\bhunger\b|\blethargy\b|\binsulin\b|\btreatment\b|\bcomplications\b|\bkidney\b|\bheart disease\b|\bstroke\b|\bblindness\b", sentence, re.IGNORECASE):
                answer.append(sentence)
        if len(answer) < 3:
            answer = sentences[:5]
    if not answer:
        answer = sentences[:5] if len(sentences) >= 5 else sentences
    return " ".join(dict.fromkeys(answer))


def clean_answer(answer):
    if not answer:
        return ""
    answer = re.split(r"\bResources\b", answer, flags=re.IGNORECASE)[0]
    answer = re.split(r"\bContext\b", answer, flags=re.IGNORECASE)[0]
    answer = re.sub(r"^The following is.*?\.?\s*", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\s+", " ", answer).strip()
    return answer


def generate_answer(llm, prompt_text):
    try:
        return clean_answer(llm.invoke(prompt_text).strip())
    except Exception:
        return ""


def main():
    st.set_page_config(page_title="Ask Chatbot!", page_icon="💬", layout="wide")
    st.title("Ask Chatbot!")

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
            retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
            docs = retrieve_documents(retriever, prompt)

            if not docs:
                result_to_show = "Sorry, I couldn't find any relevant medical information in the knowledge base."
            else:
                result_to_show = answer_from_docs(docs, prompt)

        st.chat_message("assistant").markdown(result_to_show)
        st.session_state.messages.append({"role": "assistant", "content": result_to_show})
    except Exception as e:
        with open("medibot_error.log", "a", encoding="utf-8") as f:
            f.write("OUTER EXCEPTION:\n")
            traceback.print_exception(type(e), e, e.__traceback__, file=f)
            f.write("\n")
        st.error(f"Error: {type(e).__name__}: {e}")
        result_to_show = "Sorry, I couldn't generate an answer for your question."
        st.chat_message("assistant").markdown(result_to_show)
        st.session_state.messages.append({"role": "assistant", "content": result_to_show})


if __name__ == "__main__":
    main()
