import logging
import os

from dotenv import load_dotenv, find_dotenv

from gemini_service import GeminiGenerationError, NO_CONTEXT_MESSAGE, format_sources, generate_gemini_answer
from retrieval.loaders import build_hybrid_retriever

load_dotenv(find_dotenv())

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    hybrid_retriever = build_hybrid_retriever()

    user_query = input("Write Query Here: ")
    docs = hybrid_retriever.retrieve(user_query)

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
