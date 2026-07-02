import contextlib
import os
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
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"
LLM_MODEL_ID = "gpt2"
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, say that you don't know. Do not make up an answer.
Only use the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""


@st.cache_resource
def get_vectorstore():
    try:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"local_files_only": True},
            )
    except Exception:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

    return FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True,
    )


@st.cache_resource
def load_llm():
    return HuggingFacePipeline.from_model_id(
        model_id=LLM_MODEL_ID,
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 512},
    )


def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"],
    )


def retrieve_documents(retriever, query):
    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)
    return retriever.get_relevant_documents(query)


def fallback_answer(docs):
    snippets = []
    for doc in docs[:3]:
        text = doc.page_content.strip()
        if text:
            snippets.append(text[:600])
    return "Based on the medical encyclopedia:\n\n" + "\n\n".join(snippets)


def main():
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
            vectorstore = get_vectorstore()
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            docs = retrieve_documents(retriever, prompt)

            if not docs:
                result_to_show = (
                    "Sorry, I couldn't find any relevant information in the document."
                )
            else:
                try:
                    llm = load_llm()
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        retriever=retriever,
                        chain_type="stuff",
                        chain_type_kwargs={
                            "prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
                        },
                    )
                    response = qa_chain.invoke({"query": prompt})
                    result_to_show = response["result"].strip()
                except Exception as inner_exc:
                    with open("medibot_error.log", "a", encoding="utf-8") as f:
                        f.write("INNER EXCEPTION:\n")
                        traceback.print_exception(type(inner_exc), inner_exc, inner_exc.__traceback__, file=f)
                        f.write("\n")
                    result_to_show = fallback_answer(docs)

        st.chat_message("assistant").markdown(result_to_show)
        st.session_state.messages.append(
            {"role": "assistant", "content": result_to_show}
        )

    except Exception as e:
        with open("medibot_error.log", "a", encoding="utf-8") as f:
            f.write("OUTER EXCEPTION:\n")
            traceback.print_exception(type(e), e, e.__traceback__, file=f)
            f.write("\n")
        st.error(f"Error: {type(e).__name__}: {e}")
        result_to_show = "Sorry, I couldn't generate an answer for your question."
        st.chat_message("assistant").markdown(result_to_show)
        st.session_state.messages.append(
            {"role": "assistant", "content": result_to_show}
        )


if __name__ == "__main__":
    main()
