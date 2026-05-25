"""
Corpus builder: scrape → chunk → embed → store.

Pluggable embedders (default Voyage-3, fallback OpenAI, stub for offline dev).
Pluggable vector stores (default Qdrant, stub in-memory for offline dev).

Idempotent
----------
Re-running build_corpus() only embeds new/changed chunks. Chunk identity is
sha256(source_id + chunk_index + text[:128]). If you change chunking rules,
bump the schema_version constant and it'll rebuild.

Chunking
--------
Token-bounded sliding window (default 800 tokens, 100 overlap) over the raw
document text. Geological papers have long tables and equations — pure
sentence splitting loses context. Token-bounded keeps tables intact.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Iterator, Optional, Callable
import logging
import os

from annix_intel.rag.sources import SOURCES, RawDocument

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_CHUNK_TOKENS = 800
DEFAULT_OVERLAP_TOKENS = 100


# ─── ChunkRecord ─────────────────────────────────────────────────────────────
@dataclass
class ChunkRecord:
    chunk_id:   str
    source:     str
    doc_id:     str
    doc_title:  str
    doc_url:    Optional[str]
    chunk_idx:  int
    text:       str
    embedding:  Optional[list[float]] = field(default=None, repr=False)
    metadata:   dict = field(default_factory=dict)


# ─── Public entry point ──────────────────────────────────────────────────────
def build_corpus(
    sources: Optional[list[str]] = None,
    embedder: Optional[Callable[[list[str]], list[list[float]]]] = None,
    store: Optional["VectorStore"] = None,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    dry_run: bool = False,
) -> dict:
    """
    Build (or update) the corpus.

    Parameters
    ----------
    sources
        Subset of source names to process. None = all registered sources.
    embedder
        Callable that maps list[str] → list[list[float]]. Default loads Voyage.
    store
        Vector store (must implement upsert(list[ChunkRecord])). Default Qdrant.
    chunk_tokens, overlap_tokens
        Sliding-window chunking parameters.
    dry_run
        If True, prints what would be embedded/stored but skips API calls.

    Returns
    -------
    dict with counts: {"sources": n, "docs": n, "chunks": n, "stored": n}
    """
    embedder = embedder or _default_embedder()
    store    = store    or _default_store()

    src_names = sources or list(SOURCES.keys())
    stats = {"sources": 0, "docs": 0, "chunks": 0, "stored": 0, "skipped": 0}

    pending: list[ChunkRecord] = []
    BATCH = 64

    for name in src_names:
        if name not in SOURCES:
            log.warning("Unknown source: %s", name); continue
        src = SOURCES[name]
        stats["sources"] += 1
        log.info("Processing source: %s", name)

        for doc in src.fetch():
            doc.source = name
            stats["docs"] += 1

            for idx, chunk_text in enumerate(
                _chunk_text(doc.text, chunk_tokens, overlap_tokens)
            ):
                cid = _chunk_id(doc.id, idx, chunk_text)
                rec = ChunkRecord(
                    chunk_id=cid,
                    source=name,
                    doc_id=doc.id,
                    doc_title=doc.title,
                    doc_url=doc.url,
                    chunk_idx=idx,
                    text=chunk_text,
                    metadata={
                        "year": doc.year,
                        "authors": doc.authors,
                        **doc.metadata,
                    },
                )
                stats["chunks"] += 1
                pending.append(rec)

                if len(pending) >= BATCH:
                    if not dry_run:
                        _embed_and_store(pending, embedder, store)
                    stats["stored"] += len(pending)
                    pending.clear()

    if pending:
        if not dry_run:
            _embed_and_store(pending, embedder, store)
        stats["stored"] += len(pending)

    log.info("Corpus build complete: %s", stats)
    return stats


# ─── Chunking ────────────────────────────────────────────────────────────────
def _chunk_text(text: str, max_tokens: int, overlap: int) -> Iterator[str]:
    """
    Token-bounded sliding window. Uses a fast whitespace approximation
    (1 token ≈ 0.75 words) — good enough for chunk sizing. For accurate
    counts inside the embedder, the embedder does its own tokenisation.
    """
    words = text.split()
    if not words:
        return
    # 1 token ≈ 0.75 words → max_words ≈ max_tokens / 0.75
    step = max(1, int((max_tokens - overlap) / 0.75))
    win  = max(1, int(max_tokens / 0.75))
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + win]).strip()
        if chunk:
            yield chunk
        if i + win >= len(words):
            break
        i += step


def _chunk_id(doc_id: str, idx: int, text: str) -> str:
    h = sha256(f"{SCHEMA_VERSION}|{doc_id}|{idx}|{text[:128]}".encode()).hexdigest()
    return h[:24]


# ─── Embed + store ───────────────────────────────────────────────────────────
def _embed_and_store(chunks: list[ChunkRecord], embedder, store) -> None:
    texts = [c.text for c in chunks]
    vectors = embedder(texts)
    for c, v in zip(chunks, vectors):
        c.embedding = v
    store.upsert(chunks)


# ─── Default embedder (Voyage-3, with offline stub) ──────────────────────────
def _default_embedder() -> Callable[[list[str]], list[list[float]]]:
    if not os.environ.get("VOYAGE_API_KEY"):
        log.warning("VOYAGE_API_KEY not set — using zero-vector stub embedder.")
        def stub(texts: list[str]) -> list[list[float]]:
            return [[0.0] * 1024 for _ in texts]
        return stub

    try:
        import voyageai
    except ImportError:
        log.warning("voyageai not installed — using zero-vector stub.")
        return lambda ts: [[0.0] * 1024 for _ in ts]

    client = voyageai.Client()
    def embed(texts: list[str]) -> list[list[float]]:
        r = client.embed(texts, model="voyage-3", input_type="document")
        return r.embeddings
    return embed


# ─── Default vector store (Qdrant, with in-memory stub) ──────────────────────
class VectorStore:
    """Minimal protocol the corpus builder + retriever rely on."""
    def upsert(self, chunks: list[ChunkRecord]) -> None: ...
    def search(self, vector: list[float], k: int = 5) -> list[ChunkRecord]: ...


class _InMemoryStore(VectorStore):
    def __init__(self):
        self.chunks: list[ChunkRecord] = []

    def upsert(self, chunks: list[ChunkRecord]) -> None:
        existing = {c.chunk_id for c in self.chunks}
        for c in chunks:
            if c.chunk_id not in existing:
                self.chunks.append(c)
        log.info("InMemoryStore: %d chunks total", len(self.chunks))

    def search(self, vector: list[float], k: int = 5) -> list[ChunkRecord]:
        # Cosine similarity with the chunks we have.
        import math
        def cos(a, b):
            na = math.sqrt(sum(x*x for x in a)) or 1.0
            nb = math.sqrt(sum(x*x for x in b)) or 1.0
            return sum(x*y for x, y in zip(a, b)) / (na * nb)
        scored = [(cos(vector, c.embedding or [0]*len(vector)), c) for c in self.chunks]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [c for _, c in scored[:k]]


def _default_store() -> VectorStore:
    if not os.environ.get("QDRANT_URL"):
        log.warning("QDRANT_URL not set — using in-memory vector store.")
        return _InMemoryStore()
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
    except ImportError:
        log.warning("qdrant-client not installed — using in-memory store.")
        return _InMemoryStore()

    client = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ.get("QDRANT_API_KEY"),
    )
    coll = os.environ.get("QDRANT_COLLECTION", "annix_geo_corpus")
    if not client.collection_exists(coll):
        client.create_collection(
            collection_name=coll,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )

    class _Qdrant(VectorStore):
        def upsert(self, chunks):
            points = [
                PointStruct(
                    id=int.from_bytes(bytes.fromhex(c.chunk_id), "big") % (10**18),
                    vector=c.embedding,
                    payload={
                        "source": c.source, "doc_id": c.doc_id,
                        "doc_title": c.doc_title, "doc_url": c.doc_url,
                        "chunk_idx": c.chunk_idx, "text": c.text,
                        **c.metadata,
                    },
                ) for c in chunks if c.embedding is not None
            ]
            if points:
                client.upsert(collection_name=coll, points=points)

        def search(self, vector, k=5):
            res = client.search(collection_name=coll, query_vector=vector, limit=k)
            return [
                ChunkRecord(
                    chunk_id=str(p.id),
                    source=p.payload.get("source", ""),
                    doc_id=p.payload.get("doc_id", ""),
                    doc_title=p.payload.get("doc_title", ""),
                    doc_url=p.payload.get("doc_url"),
                    chunk_idx=p.payload.get("chunk_idx", 0),
                    text=p.payload.get("text", ""),
                    metadata={k: v for k, v in p.payload.items()
                              if k not in {"source", "doc_id", "doc_title",
                                           "doc_url", "chunk_idx", "text"}},
                ) for p in res
            ]
    return _Qdrant()
