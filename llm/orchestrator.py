"""
The orchestrator: a Claude conversation with tool_use enabled, plus optional
RAG context retrieval before each call. This is what produces the dossier.

Usage
-----
    from annix_intel.llm.orchestrator import evaluate_claim_block

    result = evaluate_claim_block(
        bbox=(-114.5, 53.0, -114.0, 53.5),
        question="Are there structural traps or fault intersections in this block "
                 "that the operator's published targets might be missing?",
        commodity="hydrogen",          # or "lithium" / "uranium"
        max_iters=8,                   # tool-use loop limit
    )
    print(result["text"])
    for cite in result["citations"]:
        print("  -", cite)

This module degrades gracefully — if ANTHROPIC_API_KEY is missing, it returns
a stub response with the tool calls it *would* have made, useful for offline dev.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

from annix_intel.llm.tools import TOOL_DEFINITIONS, run_tool

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-5"
SYSTEM_PROMPT = """You are Annix Geo, a subsurface intelligence system used by
exploration geologists to evaluate hydrogen and critical-mineral claim blocks.

You have tools to query the Saskatchewan Mineral Deposits Index, the Alberta
Geological Survey, and to parse customer well-log files. Use them aggressively
— make multiple tool calls when needed. Never invent data; if a tool returns
no results, say so.

Every claim you make in your final answer must be supported by either:
  (a) a tool result you got in this conversation, or
  (b) a retrieved passage from the geological literature (when RAG context
      is provided in the user message).

Structure the final answer as:
  1. **Verdict** — one paragraph, decision-grade.
  2. **Evidence** — bulleted, each bullet ending with [source].
  3. **What's missing** — what data would tighten the verdict.
"""


@dataclass
class DossierResult:
    text: str
    citations: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    model: str = ""
    finish_reason: str = ""


# ─── Main entry point ────────────────────────────────────────────────────────
def evaluate_claim_block(
    question: str,
    bbox: tuple[float, float, float, float] | None = None,
    commodity: str | None = None,
    rag_context: list[str] | None = None,
    max_iters: int = 6,
    model: str = DEFAULT_MODEL,
) -> DossierResult:
    """
    Run Claude in tool-use mode against the ingestion connectors to produce
    a dossier-grade evaluation.

    Parameters
    ----------
    question
        The geological question. Be specific.
    bbox
        Optional bbox (WGS84) to scope all spatial queries.
    commodity
        Optional commodity hint to bias tool selection.
    rag_context
        Pre-retrieved literature snippets to inject as a system message
        addendum. Populated by annix_intel.rag.retrieve.search().
    max_iters
        Maximum tool-use turns before forcing an answer.
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        return _stub_response(question, "anthropic SDK not installed")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _stub_response(question, "ANTHROPIC_API_KEY not set")

    client = Anthropic(api_key=api_key)

    # ── Build the opening user message ──────────────────────────────────────
    user_payload = {"question": question}
    if bbox:
        user_payload["bbox_wgs84"] = list(bbox)
    if commodity:
        user_payload["commodity"] = commodity

    user_msg = "GEOLOGICAL EVALUATION REQUEST\n" + json.dumps(user_payload, indent=2)
    if rag_context:
        user_msg += (
            "\n\nRELEVANT LITERATURE (use these as citations where applicable):\n"
            + "\n\n---\n\n".join(rag_context[:8])
        )

    messages: list[dict] = [{"role": "user", "content": user_msg}]
    tool_log:  list[dict] = []
    citations: list[str]  = []

    for _it in range(max_iters):
        resp = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        if resp.stop_reason != "tool_use":
            # Final assistant turn — extract text and citations.
            text = _extract_text(resp.content)
            citations = _extract_citations(text)
            return DossierResult(
                text=text,
                citations=citations,
                tool_calls=tool_log,
                model=model,
                finish_reason=resp.stop_reason,
            )

        # Run every requested tool, append results, loop.
        messages.append({"role": "assistant", "content": resp.content})
        tool_results = []
        for block in resp.content:
            if block.type == "tool_use":
                result = run_tool(block.name, block.input)
                tool_log.append({
                    "tool": block.name,
                    "input": block.input,
                    "ok": result.get("ok", False),
                })
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=str),
                })
        messages.append({"role": "user", "content": tool_results})

    # Out of iterations — force a final answer.
    messages.append({
        "role": "user",
        "content": "Stop using tools and give your final verdict with current information.",
    })
    final = client.messages.create(
        model=model, max_tokens=4096, system=SYSTEM_PROMPT, messages=messages,
    )
    text = _extract_text(final.content)
    return DossierResult(
        text=text,
        citations=_extract_citations(text),
        tool_calls=tool_log,
        model=model,
        finish_reason="max_iters",
    )


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _extract_text(content_blocks) -> str:
    return "\n".join(
        b.text for b in content_blocks if getattr(b, "type", None) == "text"
    )


def _extract_citations(text: str) -> list[str]:
    """
    Naive citation extractor — pulls anything in square brackets. Replace
    with a proper structured-output schema once you wire that in.
    """
    import re
    return sorted(set(re.findall(r"\[([^\]]+)\]", text)))


def _stub_response(question: str, reason: str) -> DossierResult:
    """Useful for offline development without burning API calls."""
    return DossierResult(
        text=(
            f"[stub] Would evaluate: {question[:120]}...\n"
            f"Skipped Anthropic call: {reason}.\n"
            f"Available tools: {[t['name'] for t in TOOL_DEFINITIONS]}"
        ),
        citations=[],
        tool_calls=[],
        finish_reason="stub",
    )
