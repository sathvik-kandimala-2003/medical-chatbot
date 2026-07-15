import logging
import os

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)

# NOTE: the frozen snapshot id "gemini-2.5-flash" returns 404 "no longer
# available to new users" for this API key (a Google-side deprecation of that
# specific snapshot, unrelated to this app). "gemini-flash-latest" resolves
# to "gemini-3.5-flash" under the hood, whose free-tier quota
# (20 requests/day) this key exhausted during testing. "gemini-flash-lite-latest"
# is a separate model with its own quota bucket and was confirmed working
# live when the other model got rate-limited. If you hit 429s again, check
# https://ai.dev/rate-limit and swap this to whichever model currently has
# quota, or upgrade the API key's billing tier.
GEMINI_MODEL_NAME = "gemini-flash-lite-latest"

NO_CONTEXT_MESSAGE = "I could not find enough information in the uploaded medical documents."

# Single source of truth for the prompt sent to Gemini. Nothing else in the
# app should build its own prompt string.
MEDICAL_ASSISTANT_PROMPT = """You are a professional medical information assistant. You help users \
understand medical topics using ONLY the reference material provided below.

Rules:
1. Answer using ONLY the information inside "Context". Never use outside knowledge and never guess.
2. Never hallucinate or invent medical facts, numbers, or details that are not in the context.
3. If the context does not contain enough information to answer the question, reply with EXACTLY \
this sentence and nothing else: "{no_context_message}"
4. Do not copy long passages verbatim. Summarize and explain concepts in simple, clear language.
5. Remove repeated information instead of restating it.
6. Keep the answer concise: about 150-250 words, unless the user explicitly asks for more detail.
7. Format the answer with markdown headings and bullet points where appropriate. Only use headings \
from this list, only when the context actually supports that section, and only in this order:
## Definition
## Symptoms
## Causes
## Diagnosis
## Treatment
## Prevention
## Complications
Do not invent a section that isn't supported by the context. Do not add a "Sources" section \
yourself - it is added separately.

Context:
{context}

Question:
{question}

Answer:
"""


class GeminiGenerationError(Exception):
    """Raised when Gemini cannot produce an answer (missing key, API failure, empty response)."""


def get_gemini_model_name():
    return GEMINI_MODEL_NAME


def _format_context(context_docs):
    excerpts = [
        f"[Excerpt {i}]\n{doc.page_content.strip()}"
        for i, doc in enumerate(context_docs, start=1)
    ]
    return "\n\n".join(excerpts)


def format_sources(context_docs):
    """Build a deterministic Sources block from retrieved Document metadata.

    Built from metadata only (never from LLM output) so citations can't be fabricated.
    """
    seen = []
    for doc in context_docs:
        source = os.path.basename(doc.metadata.get("source") or "Unknown document")
        page_label = doc.metadata.get("page_label")
        if not page_label:
            page = doc.metadata.get("page")
            page_label = str(page + 1) if isinstance(page, int) else None
        entry = (source, page_label)
        if entry not in seen:
            seen.append(entry)

    if not seen:
        return ""

    lines = ["### Sources"]
    for source, page_label in seen:
        lines.append(f"- {source} (page {page_label})" if page_label else f"- {source}")
    return "\n".join(lines)


def generate_gemini_answer(question, context_docs, temperature=0.2):
    """Generate an answer from Gemini using the raw retrieved Document chunks.

    Raises GeminiGenerationError on any failure (missing key, API error, empty
    response) so callers can surface a real error instead of silently
    falling back to something else.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise GeminiGenerationError("GEMINI_API_KEY is not configured.")

    context = _format_context(context_docs)
    prompt = MEDICAL_ASSISTANT_PROMPT.format(
        no_context_message=NO_CONTEXT_MESSAGE,
        context=context,
        question=question,
    )

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME)
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                # gemini-flash-latest is a "thinking" model: its internal
                # reasoning tokens are deducted from max_output_tokens before
                # the visible answer is written. This SDK version
                # (google-generativeai==0.8.0) has no thinking_config knob to
                # disable that, so the budget must be large enough to cover
                # both the reasoning pass and a ~150-250 word answer.
                "max_output_tokens": 4096,
            },
        )
        text = getattr(response, "text", "").strip()
    except Exception as exc:
        logger.exception("Gemini API call failed")
        raise GeminiGenerationError(f"Gemini API call failed: {exc}") from exc

    if not text:
        raise GeminiGenerationError("Gemini returned an empty response.")
    return text
