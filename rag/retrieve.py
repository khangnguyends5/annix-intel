"""
Retrieval API: search(question, k) → list[RetrievedPassage].

What the orchestrator calls before invoking Claude. Wraps:
    1. Question embedding (same model as corpus builder)
    2. Vector store search
    3. Optional re-ranking
    4. Returns formatted snippets ready to splice into the prompt
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import os
import logging

from annix_intel.rag.corpus import _default_embedder, _default_store, VectorStore

log = logging.getLogger(__name__)


@dataclass
class RetrievedPassage:
    text:      str
    source:    str
    doc_title: str
    doc_url:   Optional[str]
    score:     Optional[float] = None

    def for_prompt(self) -> str:
        """Format ready to splice into a Claude system / user message."""
        head = f"[{self.source} — {self.doc_title}]"
        if self.doc_url:
            head += f" ({self.doc_url})"
        return f"{head}\n{self.text}"


def search(
    question: str,
    k: int = 6,
    store: Optional[VectorStore] = None,
    embedder=None,
    rerank: bool = True,
) -> list[RetrievedPassage]:
    """
    Run a single retrieval. Reuses the same embedder/store the corpus was
    built with — both are determined at runtime by env vars.

    Re-ranking
    ----------
    If `rerank=True` and Voyage's reranker is available, the top 3*k
    candidates are scored against the question and the top-k returned.
    Significantly better precision than vector-similarity alone.
    """
    embedder = embedder or _default_embedder()
    store    = store    or _default_store()

    q_vec = embedder([question])[0]
    candidates = store.search(q_vec, k=k * 3 if rerank else k)

    if rerank and len(candidates) > k:
        candidates = _rerank(question, candidates, top_k=k)

    return [
        RetrievedPassage(
            text=c.text,
            source=c.source,
            doc_title=c.doc_title,
            doc_url=c.doc_url,
        ) for c in candidates[:k]
    ]


def _rerank(question, candidates, top_k):
    if not os.environ.get("VOYAGE_API_KEY"):
        return candidates[:top_k]
    try:
        import voyageai
    except ImportError:
        return candidates[:top_k]
    client = voyageai.Client()
    try:
        r = client.rerank(
            question, [c.text for c in candidates], model="rerank-2", top_k=top_k,
        )
    except Exception as e:                                          # noqa: BLE001
        log.warning("Rerank failed (%s) — falling back to vector order.", e)
        return candidates[:top_k]
    return [candidates[item.index] for item in r.results]
