"""Ground-truth schema for offline retrieval evaluation.

A question only needs *some* of the relevance signals to be usable -
evaluation.metrics.relevance.is_relevant() treats relevant_documents,
relevant_chunks, and expected_sources as alternative (OR'd) ways to judge a
retrieved chunk relevant, since hand-authoring exact chunk boundaries is
impractical but naming the right source document/page is easy.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class QAItem:
    question: str
    ground_truth_answer: Optional[str] = None
    # Source filenames (basename) that are considered relevant, e.g. "The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"
    relevant_documents: List[str] = field(default_factory=list)
    # Snippets of ground-truth chunk text; matched by substring/word-overlap, not exact chunk ID
    relevant_chunks: List[str] = field(default_factory=list)
    # "filename.pdf" or "filename.pdf#page=123" - page-qualified relevance
    expected_sources: List[str] = field(default_factory=list)
