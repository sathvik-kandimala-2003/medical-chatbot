"""Relevance judgment: is a retrieved chunk relevant to a dataset question?

Three ground-truth signals, any one is sufficient (OR'd together) - see
QAItem's docstring for why: exact chunk IDs aren't practical to hand-author,
but naming the right source document (and optionally page) is easy, and a
short excerpt of the expected text is a reasonable middle ground.

  1. Document match: chunk's source file is in relevant_documents.
  2. Source(+page) match: "file.pdf" or "file.pdf#page=N" in expected_sources.
  3. Chunk-text match: chunk content contains (or heavily overlaps) an
     excerpt in relevant_chunks.
"""

import os
from typing import Iterable

from langchain_core.documents import Document

from evaluation.datasets.schema import QAItem
from evaluation.utils.text import normalize_text


def _source_matches(doc_source: str, candidates: Iterable[str]) -> bool:
    doc_basename = os.path.basename(doc_source or "")
    return any(os.path.basename(candidate.split("#")[0]) == doc_basename for candidate in candidates)


def _page_qualified_match(doc_source: str, doc_page, candidates: Iterable[str]) -> bool:
    doc_basename = os.path.basename(doc_source or "")
    doc_page_str = str(doc_page) if doc_page is not None else None
    for candidate in candidates:
        source_part, _, page_part = candidate.partition("#page=")
        if os.path.basename(source_part) != doc_basename:
            continue
        if not page_part:
            return True  # source named without a page qualifier -> whole document is relevant
        if doc_page_str is not None and page_part.strip() == doc_page_str.strip():
            return True
    return False


def _chunk_text_matches(doc_text: str, relevant_chunks: Iterable[str], min_overlap_ratio: float) -> bool:
    normalized_doc = normalize_text(doc_text)
    doc_words = set(normalized_doc.split())
    for excerpt in relevant_chunks:
        needle = normalize_text(excerpt)
        if not needle:
            continue
        if needle in normalized_doc:
            return True
        needle_words = set(needle.split())
        if needle_words and len(needle_words & doc_words) / len(needle_words) >= min_overlap_ratio:
            return True
    return False


def is_relevant(doc: Document, qa_item: QAItem, min_overlap_ratio: float = 0.6) -> bool:
    metadata = doc.metadata or {}
    source = metadata.get("source", "")
    page = metadata.get("page")
    page_label = str(page) if isinstance(page, int) else metadata.get("page_label")

    if qa_item.relevant_documents and _source_matches(source, qa_item.relevant_documents):
        return True

    if qa_item.expected_sources and _page_qualified_match(source, page_label, qa_item.expected_sources):
        return True

    if qa_item.relevant_chunks and _chunk_text_matches(doc.page_content, qa_item.relevant_chunks, min_overlap_ratio):
        return True

    return False
