"""Per-query metrics collected across the retrieval + generation pipeline."""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class QueryMetrics:
    embedding_time_ms: float = 0.0
    vector_search_time_ms: float = 0.0
    bm25_search_time_ms: float = 0.0
    fusion_time_ms: float = 0.0
    reranking_time_ms: float = 0.0
    llm_generation_time_ms: float = 0.0
    total_response_time_ms: float = 0.0

    # Candidate pool fused from FAISS + BM25, before reranking (up to fusion_top_n).
    retrieved_documents_count: int = 0
    # What actually made it into the Gemini prompt (post-rerank, up to rerank_top_n).
    final_context_chunk_count: int = 0
    final_context_chars: int = 0

    estimated_prompt_tokens: int = 0
    estimated_output_tokens: int = 0

    def as_display_rows(self) -> List[Tuple[str, str]]:
        """Ordered (label, formatted value) pairs for UI rendering."""
        return [
            ("Embedding Time", f"{self.embedding_time_ms:.1f} ms"),
            ("Vector Search Time", f"{self.vector_search_time_ms:.1f} ms"),
            ("BM25 Search Time", f"{self.bm25_search_time_ms:.1f} ms"),
            ("Fusion Time", f"{self.fusion_time_ms:.1f} ms"),
            ("Reranking Time", f"{self.reranking_time_ms:.1f} ms"),
            ("LLM Generation Time", f"{self.llm_generation_time_ms:.1f} ms"),
            ("Total Response Time", f"{self.total_response_time_ms:.1f} ms"),
            ("Retrieved Documents Count", str(self.retrieved_documents_count)),
            ("Final Context Chunks", str(self.final_context_chunk_count)),
            ("Final Context Size", f"{self.final_context_chars:,} chars"),
            ("Estimated Prompt Tokens", str(self.estimated_prompt_tokens)),
            ("Estimated Output Tokens", str(self.estimated_output_tokens)),
        ]
