from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from gemini_service import GeminiGenerationError, NO_CONTEXT_MESSAGE, format_sources, generate_gemini_answer

load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"
RETRIEVAL_K = 4


def main():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

    user_query = input("Write Query Here: ")
    docs = db.as_retriever(search_kwargs={"k": RETRIEVAL_K}).invoke(user_query)

    if not docs:
        print(NO_CONTEXT_MESSAGE)
        return

    try:
        answer = generate_gemini_answer(user_query, docs)
    except GeminiGenerationError as exc:
        print(f"LLM unavailable: {exc}")
        return

    print("RESULT:\n", answer)
    sources = format_sources(docs)
    if sources:
        print("\n" + sources)


if __name__ == "__main__":
    main()
