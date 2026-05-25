# annix_intel

Shared intelligence layer for the **Annix Geo** product family — currently
**Annix Geo Minerals** (Saskatchewan critical-mineral targeting) and
**Annix Geo H2**
(natural hydrogen). One ingestion pipeline, one Claude-orchestrated brain,
one RAG corpus — used by both products.

```
annix_intel/
├── ingest/         Data connectors (LAS, AGS, SMDI, ...)
├── llm/            Claude tool definitions + orchestrator
├── rag/            Retrieval over geological literature
└── tests/
```

## Why this package exists

The two products were heading toward duplicating the same three things:

1. Data ingestion (LAS, SMDI, AGS, well headers, formation tops)
2. LLM orchestration with grounded citations
3. A geological-literature RAG corpus

Building them twice is wasteful. `annix_intel` is the single dependency both
import. Each connector emits canonical types — `WellLog`, `DepositRecord`,
`GeologicLayer` — so downstream code never cares which source the data came from.

## Quick smoke test

```bash
pip install -r requirements.txt
python -c "from annix_intel.llm.tools import TOOL_DEFINITIONS; print([t['name'] for t in TOOL_DEFINITIONS])"
```

## Layer 1 — Ingestion

Each connector returns a canonical type. Adding a new source = one file.

| Connector | Function | Returns | Status |
|---|---|---|---|
| LAS file reader | `read_las(path)` | `WellLog` | ✓ working |
| Alberta GS wells | `fetch_ags_wells(formation=, bbox=)` | `list[WellLog]` | ✓ working |
| AGS formation layer | `fetch_ags_formation_tops(formation, bbox)` | `GeologicLayer` | ✓ working |
| Saskatchewan SMDI | `fetch_smdi(commodity=, bbox=)` | `list[DepositRecord]` | ✓ working |
| SEG-Y seismic | `read_segy(path)` | `GeologicLayer` | TODO |
| USGS NWIS groundwater | `fetch_nwis(...)` | `list[WellLog]` (chemistry curves) | TODO |
| BC / MB / Ontario surveys | `fetch_bcgs / fetch_mgs / fetch_ogs` | `list[DepositRecord]` | TODO |

The pattern for any new connector:

```python
# annix_intel/ingest/mynewsource.py
from annix_intel.ingest.types import WellLog

def fetch_my_source(...) -> list[WellLog]:
    ...
```

Then register it in `annix_intel/ingest/__init__.py` and add a Claude tool
definition in `annix_intel/llm/tools.py`.

## Layer 2 — LLM orchestrator

`evaluate_claim_block()` runs a Claude conversation in tool_use mode:

```python
from annix_intel.llm.orchestrator import evaluate_claim_block
from annix_intel.rag.retrieve import search

# Optional: pre-fetch literature context
ctx = [p.for_prompt() for p in search(
    "Devonian seal integrity above Duperow Formation",
    k=6,
)]

result = evaluate_claim_block(
    question=(
        "The operator has a proposed well at 53.42N, 114.18W targeting "
        "the Leduc reef. Is there a deeper Beaverhill Lake target they're "
        "missing? Check structural traps and fluid pathways in this block."
    ),
    bbox=(-114.5, 53.2, -113.8, 53.6),
    commodity="hydrogen",
    rag_context=ctx,
)

print(result.text)
print("Citations:", result.citations)
print("Tool calls:", [t["tool"] for t in result.tool_calls])
```

If `ANTHROPIC_API_KEY` isn't set, returns a stub describing what *would* have
run — useful for offline dev.

## Layer 3 — RAG corpus

The defensible moat. Off-the-shelf LLMs know zero WCSB stratigraphy. A
500-document corpus over AGS bulletins + USGS open-file reports + natural-H2
papers makes Claude geologically literate in your domain.

### Build the corpus

```python
from annix_intel.rag import build_corpus

# Build everything you have connectors for
stats = build_corpus()        # {'sources': 6, 'docs': 412, 'chunks': 8841, ...}

# Or a single source
stats = build_corpus(sources=["local_pdfs"])
```

### Query it

```python
from annix_intel.rag import search

passages = search("seal integrity Devonian evaporite Duperow", k=6)
for p in passages:
    print(p.for_prompt())
```

### Eval it

```bash
python -m annix_intel.rag.eval rag_eval.yaml
```

Eval cases are YAML — see `annix_intel/rag/eval.py` for the schema. Output
written to `rag_eval_report.json`; CI exits 1 if any case fails.

## 30-day corpus build plan

The package is wired up; the corpus content is the work. Priority order:

### Week 1 — local PDFs + AGS bulletins

- Drop every WCSB paper / report you already own into `./data/rag_inputs/`.
  `local_pdfs` source picks them up automatically. Target: 50–100 docs.
- Scrape the AGS Open File Reports index (≈2000 PDFs, all public). Implement
  `fetch_ags_open_file_reports()` in `annix_intel/rag/sources.py`.
- Run `build_corpus(sources=["local_pdfs", "ags_open_file_reports"])`.

### Week 2 — USGS + natural hydrogen literature

- USGS Open-File Reports: implement `fetch_usgs_open_file_reports()`.
  Filter to the ~200 relevant ones (groundwater chemistry, hydrogen, basement
  geology, redox geochemistry).
- Natural hydrogen papers: maintain a hand-curated list of DOIs. The
  literature is small (<200 papers globally) and worth manual curation.
  Implement `fetch_natural_h2_papers()` reading from `data/nh2_dois.txt`.

### Week 3 — eval harness + iteration

- Write 30 eval cases in `rag_eval.yaml` covering:
  - WCSB stratigraphy (Mannville, Beaverhill Lake, Leduc, Duperow, basement)
  - Natural-H2 geochemistry (serpentinization, sulfate reduction signatures)
  - Specific operators' published targets (your Blind Spot dossier candidates)
- Run eval. Recall@6 should hit 80%+ on first pass with a 400-doc corpus.
- Fix retrieval failures one of three ways: add missing docs, tune chunking,
  add Voyage reranker.

### Week 4 — production hardening

- Move from in-memory vector store to Qdrant (free tier, self-host or cloud).
  Set `QDRANT_URL`. The store swap is transparent — no code changes.
- Move from stub embedder to Voyage-3. Set `VOYAGE_API_KEY`. Re-run
  `build_corpus()` — chunk IDs are stable so only embeddings get re-computed.
- Wire `evaluate_claim_block()` into both Annix Geo Minerals and Annix Geo H2
  as the dossier generator.

## Environment variables

| Var | Default behavior if missing |
|---|---|
| `ANTHROPIC_API_KEY` | Orchestrator returns a stub response |
| `VOYAGE_API_KEY` | Embedder returns zero-vectors (corpus still chunks + stores, just useless for search) |
| `QDRANT_URL` | Falls back to in-memory store (fine for dev, lost on restart) |
| `QDRANT_API_KEY` | Used if `QDRANT_URL` points at Qdrant Cloud |
| `QDRANT_COLLECTION` | Defaults to `annix_geo_corpus` |

## How both products use this

**Annix Geo H2** — replaces the current toy `hydrochem` with PHREEQC calls
(which become a Claude tool), then uses `evaluate_claim_block()` to produce
the natural-language verdict + recommended sampling locations that go in the
dossier. RAG context primes Claude with the relevant WCSB hydrogen literature.

**Annix Geo Minerals** — replaces `04_llm_briefs.py` (Gemini/Groq) with
`evaluate_claim_block()`. The investor teasers become defensible briefs with
real citations from AGS bulletins and USGS reports.

Same package, two products under one **Annix Geo** brand. That's the point.
