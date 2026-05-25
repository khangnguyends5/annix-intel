"""
The source registry — where the geological corpus comes from.

Curated, not crawled. Random web-scraping makes a noisy corpus that doesn't
beat ChatGPT. The whole point of this RAG is high signal-to-noise WCSB-
specific material that off-the-shelf LLMs don't have.

How to extend
-------------
Either edit SOURCES below, or in your own code:

    from annix_intel.rag.sources import register_source

    register_source(
        name="my_company_internal_reports",
        kind="folder",
        path="/data/internal/exploration_reports",
        license="internal",
        priority="high",
    )

Each source has a `fetch()` callable that yields RawDocument records. The
corpus builder calls fetch() then chunks, embeds, stores.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


# ─── RawDocument: what every source.fetch() yields ───────────────────────────
@dataclass
class RawDocument:
    id:        str                       # unique within (source_name, document)
    title:     str
    text:      str                       # full plain text
    url:       str | None = None
    authors:   list[str] = field(default_factory=list)
    year:      int | None = None
    source:    str = ""                  # filled by the registry
    metadata:  dict = field(default_factory=dict)


# ─── Source spec ─────────────────────────────────────────────────────────────
@dataclass
class Source:
    name:        str
    description: str
    fetch:       Callable[[], Iterator[RawDocument]]
    priority:    str = "medium"          # "high" | "medium" | "low"
    license:     str = "unknown"         # "public-domain" | "open" | "internal" | ...


SOURCES: dict[str, Source] = {}


def register_source(name: str, **kwargs) -> Source:
    """Register a Source object. Returns it for chaining."""
    s = Source(name=name, **kwargs)
    SOURCES[name] = s
    log.info("Registered RAG source: %s (priority=%s, license=%s)",
             name, s.priority, s.license)
    return s


# ─── Built-in source: local PDF/text folder ──────────────────────────────────
def _fetch_folder(path: Path) -> Iterator[RawDocument]:
    """
    Generic local-folder source. Yields one RawDocument per file.
    PDFs are extracted with pymupdf if available; .txt and .md read raw.
    """
    for p in sorted(path.rglob("*")):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        try:
            if suffix == ".pdf":
                text = _pdf_to_text(p)
            elif suffix in (".txt", ".md"):
                text = p.read_text(encoding="utf-8", errors="replace")
            else:
                continue
        except Exception as e:                                      # noqa: BLE001
            log.warning("Failed to read %s: %s", p, e)
            continue

        if not text or len(text) < 200:
            continue

        yield RawDocument(
            id=str(p.relative_to(path)),
            title=p.stem,
            text=text,
            url=p.as_uri(),
            metadata={"path": str(p), "size_bytes": p.stat().st_size},
        )


def _pdf_to_text(path: Path) -> str:
    try:
        import fitz
        import pymupdf  # noqa: F401, fitz is the import name
    except ImportError as e:
        raise RuntimeError(
            "PDF source requires pymupdf. Install: pip install pymupdf"
        ) from e
    doc = fitz.open(path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n\n".join(pages)


# ─── Seed the registry with the high-priority targets ────────────────────────
# Each of these needs its own connector built — track in 30-day plan below.
def _todo_fetch(name: str) -> Callable[[], Iterator[RawDocument]]:
    def stub() -> Iterator[RawDocument]:
        log.warning("Source '%s' not yet implemented — skipping.", name)
        return iter(())
    return stub


register_source(
    name="ags_open_file_reports",
    description=(
        "Alberta Geological Survey Open File Reports. WCSB stratigraphy, "
        "structural geology, formation characterizations. ~2000 PDFs."
    ),
    priority="high",
    license="open",
    fetch=_todo_fetch("ags_open_file_reports"),
)

register_source(
    name="usgs_open_file_reports",
    description=(
        "USGS Open-File Reports relevant to natural hydrogen and groundwater "
        "geochemistry. Searchable index at pubs.usgs.gov."
    ),
    priority="high",
    license="public-domain",
    fetch=_todo_fetch("usgs_open_file_reports"),
)

register_source(
    name="natural_h2_papers",
    description=(
        "Curated peer-reviewed papers on natural hydrogen exploration "
        "(Mali, Russia, France, Australia, Kansas). Maintained by hand."
    ),
    priority="high",
    license="mixed",
    fetch=_todo_fetch("natural_h2_papers"),
)

register_source(
    name="ags_bulletins",
    description="AGS Bulletins — peer-reviewed monographs on Alberta geology.",
    priority="high",
    license="open",
    fetch=_todo_fetch("ags_bulletins"),
)

register_source(
    name="cspg_reservoir_journal",
    description=(
        "Canadian Society of Petroleum Geologists Reservoir journal. "
        "Requires subscription — for paying customers only."
    ),
    priority="medium",
    license="commercial",
    fetch=_todo_fetch("cspg_reservoir_journal"),
)

register_source(
    name="local_pdfs",
    description=(
        "Local folder. Drop any geological PDF or .md here and it's added "
        "to the corpus on next build. Useful for customer-supplied reports."
    ),
    priority="high",
    license="varies",
    fetch=lambda: _fetch_folder(Path("./data/rag_inputs")),
)


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"{len(SOURCES)} sources registered:\n")
    for s in SOURCES.values():
        print(f"  [{s.priority:6s}] {s.name}")
        print(f"           {s.description}")
        print(f"           license={s.license}\n")
