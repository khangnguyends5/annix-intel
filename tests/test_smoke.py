"""
Smoke tests — fast, no network, no API keys required.

These are the "did anything break?" tests. They run on every push. Real
integration tests against AGS/Anthropic/Voyage live in tests/test_live.py
and are gated by env vars + the @pytest.mark.live marker.
"""

from __future__ import annotations


def test_package_imports():
    import annix_intel
    assert annix_intel.__version__


def test_ingest_types_construct():
    from annix_intel.ingest import DepositRecord, GeologicLayer, WellLog

    w = WellLog(uwi="100/01-02-003-04W5/00")
    assert w.uwi
    assert w.summary()["uwi"] == "100/01-02-003-04W5/00"

    d = DepositRecord(id="X1", primary_commodities=["Lithium"])
    assert d.summary()["primary_commodities"] == ["Lithium"]

    g = GeologicLayer(name="t", layer_type="fault")
    assert g.crs == "EPSG:4326"


def test_tool_definitions_well_formed():
    """Every tool def must have name + description + input_schema."""
    from annix_intel.llm.tools import TOOL_DEFINITIONS

    seen_names = set()
    for t in TOOL_DEFINITIONS:
        assert "name" in t and t["name"], f"tool missing name: {t}"
        assert t["name"] not in seen_names, f"duplicate tool name: {t['name']}"
        seen_names.add(t["name"])
        assert t.get("description"), f"tool {t['name']} missing description"
        sch = t.get("input_schema") or {}
        assert sch.get("type") == "object", f"tool {t['name']} bad schema"
        assert isinstance(sch.get("properties", {}), dict)


def test_tool_dispatcher_unknown_tool():
    from annix_intel.llm.tools import run_tool
    r = run_tool("does_not_exist", {})
    assert r["ok"] is False
    assert "Unknown tool" in r["error"]


def test_tool_dispatcher_handles_exceptions():
    from annix_intel.llm.tools import run_tool
    # read_las with bogus path should NOT raise — must return ok=False
    r = run_tool("read_las_file", {"path": "/no/such/file.las"})
    assert r["ok"] is False
    assert "error" in r


def test_orchestrator_stubs_without_api_key(monkeypatch):
    """No ANTHROPIC_API_KEY → stub response, never raises."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from annix_intel.llm.orchestrator import evaluate_claim_block
    r = evaluate_claim_block("Where are H2 sources in Alberta?")
    assert r.finish_reason == "stub"
    assert "ANTHROPIC_API_KEY" in r.text or "stub" in r.text.lower()


def test_rag_sources_registered():
    from annix_intel.rag import SOURCES
    assert "local_pdfs" in SOURCES
    assert len(SOURCES) >= 5


def test_rag_build_dry_run_empty_folder(tmp_path):
    """build_corpus over the local_pdfs source — empty folder is a valid edge."""
    from annix_intel.rag import build_corpus
    # local_pdfs reads ./data/rag_inputs which won't exist in CI — that's fine.
    stats = build_corpus(sources=["local_pdfs"], dry_run=True)
    assert stats["sources"] == 1
    assert stats["docs"] == 0
    assert stats["chunks"] == 0


def test_rag_search_returns_list_even_when_empty():
    from annix_intel.rag.retrieve import search
    out = search("anything", k=3)
    assert isinstance(out, list)
    assert len(out) <= 3
