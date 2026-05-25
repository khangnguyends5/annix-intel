"""
annix_intel.rag — Retrieval over geological literature.

Three layers:

    sources    Registry of where to scrape from (AGS publications, USGS
               open-file reports, hydrogen-exploration papers, etc.).

    corpus     Build pipeline: scrape → chunk → embed → store. Idempotent.

    retrieve   Query API: search(question, k) → list[str] passages with
               source attribution. Used by annix_intel.llm.orchestrator.

    eval       Eval harness: did the corpus retrieve the right passages
               for known geological questions? Treats RAG as a tested
               component, not magic.
"""

from annix_intel.rag.sources  import SOURCES, register_source
from annix_intel.rag.corpus   import build_corpus, ChunkRecord
from annix_intel.rag.retrieve import search, RetrievedPassage

__all__ = [
    "SOURCES", "register_source",
    "build_corpus", "ChunkRecord",
    "search", "RetrievedPassage",
]
