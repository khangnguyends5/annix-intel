"""
annix_intel — Shared intelligence layer for Annix Geo and Annix Geo H2.

Three subpackages:

    annix_intel.ingest    Data connectors. LAS well logs, AGS / SMDI public
                          APIs, geospatial files. Returns canonical dataclasses.

    annix_intel.llm       Claude tool definitions + orchestrator. The brain
                          that fuses ingested data, runs physics/chemistry
                          engines, and produces dossiers.

    annix_intel.rag       Retrieval over a corpus of geological literature.
                          What lets Claude cite real WCSB studies instead of
                          hallucinating.

Import pattern:

    from annix_intel.ingest import connectors
    from annix_intel.llm import tools, orchestrator
    from annix_intel.rag import corpus, retrieve
"""

__version__ = "0.1.0"
