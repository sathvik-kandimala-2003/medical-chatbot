"""Production analytics for the chatbot: persists every interaction the
existing hybrid-retrieval pipeline already produces (QueryMetrics + ChunkTrace)
and builds session analytics / visualizations / history on top of that
historical data. Does not recompute or duplicate any retrieval, memory, or
metrics logic - see retrieval/, memory/, metrics/, gemini_service.py.
"""
