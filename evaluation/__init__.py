"""Offline retrieval evaluation for the chatbot's hybrid retrieval pipeline.

Run via `python evaluate.py` at the repo root. This package reuses
retrieval.HybridRetriever and gemini_service directly (via
retrieval.loaders.build_hybrid_retriever) rather than reimplementing any
retrieval or generation logic - see evaluator/pipeline.py.
"""
