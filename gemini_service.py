import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv, find_dotenv

from metrics.token_estimator import estimate_tokens

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
8. The Conversation History (if present) is only to help you understand what the user is \
referring to (e.g. a pronoun like "it" or "that" pointing at a topic from an earlier turn). \
Never treat the Conversation History itself as a source of medical facts - every fact in your \
answer must still come from Context.

{history}Context:
{context}

Question:
{question}

Answer:
"""

# Used to rewrite a follow-up question ("What are its symptoms?") into a
# standalone one ("What are the symptoms of diabetes?") before retrieval, so
# embedding/BM25 search see the actual topic instead of a bare pronoun.
CONDENSE_QUESTION_PROMPT = """Given the conversation history and a follow-up question, rewrite the \
follow-up question as a standalone question that includes whatever context from the history is \
needed to understand it on its own (resolve pronouns like "it"/"that"/"this" to the specific \
medical topic being discussed). If the follow-up question is already standalone, return it \
unchanged. Output ONLY the rewritten question - no preamble, no quotation marks.

Conversation History:
{history}

Follow-up Question: {question}

Standalone Question:"""


class GeminiGenerationError(Exception):
    """Raised when Gemini cannot produce an answer (missing key, API failure, empty response)."""


@dataclass
class GeminiAnswer:
    text: str
    prompt_tokens: int
    output_tokens: int


def get_gemini_model_name():
    return GEMINI_MODEL_NAME


def _format_context(context_docs):
    excerpts = [
        f"[Excerpt {i}]\n{doc.page_content.strip()}"
        for i, doc in enumerate(context_docs, start=1)
    ]
    return "\n\n".join(excerpts)


def _format_history_block(chat_history_text):
    """Render the optional Conversation History section of the prompt.

    Returns "" when there's no history, so the prompt is byte-identical to
    the pre-memory version for a first turn / callers that don't pass history.
    """
    if not chat_history_text or not chat_history_text.strip():
        return ""
    return f"Conversation History:\n{chat_history_text.strip()}\n\n"


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


def _extract_token_counts(response, prompt_text, answer_text):
    """Prefer Gemini's own reported token usage (response.usage_metadata);
    fall back to a ~4-chars/token estimate if it's unavailable on this
    response, matching the "estimated" framing in the UI.
    """
    try:
        usage = response.usage_metadata
        if usage is not None:
            return int(usage.prompt_token_count), int(usage.candidates_token_count)
    except Exception:
        logger.debug("Gemini response has no usable usage_metadata; estimating token counts")
    return estimate_tokens(prompt_text), estimate_tokens(answer_text)


def condense_question(question, chat_history_text, temperature=0.0):
    """Rewrite a follow-up question into a standalone question using recent history.

    Best-effort: if there's no history, no API key, or the call fails for any
    reason, this logs (when applicable) and returns the original question
    unchanged - a condensation failure should never block retrieval.
    """
    if not chat_history_text or not chat_history_text.strip():
        return question

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return question

    prompt = CONDENSE_QUESTION_PROMPT.format(history=chat_history_text.strip(), question=question)

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": temperature, "max_output_tokens": 256},
        )
        standalone_question = getattr(response, "text", "").strip()
    except Exception:
        logger.exception("Query condensation failed; using the original question for retrieval")
        return question

    return standalone_question or question


def generate_gemini_answer_detailed(question, context_docs, chat_history_text="", temperature=0.2):
    """Generate an answer from Gemini using the retrieved Document chunks.

    Same generation behavior as generate_gemini_answer(), but returns a
    GeminiAnswer with prompt/output token counts alongside the text, and
    accepts optional chat history so Gemini can resolve references left over
    from query condensation (e.g. "it") when composing the final answer.

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
        history=_format_history_block(chat_history_text),
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

    prompt_tokens, output_tokens = _extract_token_counts(response, prompt, text)
    return GeminiAnswer(text=text, prompt_tokens=prompt_tokens, output_tokens=output_tokens)


def generate_gemini_answer(question, context_docs, temperature=0.2):
    """Backward-compatible wrapper: same as generate_gemini_answer_detailed()
    but returns just the answer text, with no chat history in the prompt.
    """
    return generate_gemini_answer_detailed(question, context_docs, temperature=temperature).text
