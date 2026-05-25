"""
annix_intel.llm — The brain: Claude tool definitions + orchestrator.

Pattern: every ingestion connector has a matching Claude tool definition
(in `tools`). The orchestrator (in `orchestrator`) runs a Claude conversation
where Claude itself decides which tools to call to answer a geological query.

Example
-------
    from annix_intel.llm.orchestrator import evaluate_claim_block

    dossier = evaluate_claim_block(
        bbox=(-115.0, 53.0, -114.0, 54.0),   # WGS84 bbox of customer's claim
        question="Map all fluid pathways and rank deep H2 source likelihood.",
    )
    print(dossier.summary)
    for source, snippet in dossier.citations:
        print(f"  [{source}] {snippet}")
"""

from annix_intel.llm.tools import TOOL_DEFINITIONS, run_tool

__all__ = ["TOOL_DEFINITIONS", "run_tool"]
