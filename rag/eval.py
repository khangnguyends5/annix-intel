"""
Eval harness for the RAG layer.

Treats retrieval as a tested component, not magic. The eval file (YAML) lists
known geological questions and the document(s) that *should* be retrieved.
On each corpus change, we check recall@k and report regressions.

Example eval entry (rag_eval.yaml)
----------------------------------
- id: duperow_seal_quality
  question: |
    What is the seal integrity of the Devonian evaporite section above the
    Duperow Formation in southeast Saskatchewan?
  must_retrieve_any_of:
    - "AGS Bulletin 65"
    - "Anderson 2019 Duperow"
  rationale: "Standard reference for WCSB Devonian seals."

- id: natural_h2_lithology
  question: |
    Which basement lithologies are most associated with natural hydrogen
    generation via serpentinization?
  must_retrieve_any_of:
    - "Mali natural hydrogen Bourakebougou"
    - "Truche 2018 hydrogen"

Run
---
    python -m annix_intel.rag.eval rag_eval.yaml
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from annix_intel.rag.retrieve import search

log = logging.getLogger(__name__)


@dataclass
class EvalCase:
    id: str
    question: str
    must_retrieve_any_of: list[str]
    rationale: str | None = None


@dataclass
class CaseResult:
    id: str
    question: str
    passed: bool
    matched_phrase: str | None = None
    top_titles: list[str] = None
    notes: str | None = None


def load_eval_file(path: str | Path) -> list[EvalCase]:
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML required for eval files. `pip install pyyaml`") from e
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Eval file must be a YAML list of cases.")
    return [EvalCase(**case) for case in raw]


def run_eval(
    cases: list[EvalCase],
    k: int = 6,
) -> tuple[list[CaseResult], dict]:
    """
    Run every case. Returns (per-case results, summary dict).

    A case passes if at least one of `must_retrieve_any_of` substrings appears
    in the doc_title or text of any of the top-k retrieved passages.
    """
    results: list[CaseResult] = []
    for case in cases:
        passages = search(case.question, k=k)
        titles = [p.doc_title for p in passages]

        matched = None
        for phrase in case.must_retrieve_any_of:
            for p in passages:
                if phrase.lower() in p.doc_title.lower() or phrase.lower() in p.text.lower():
                    matched = phrase
                    break
            if matched:
                break

        results.append(CaseResult(
            id=case.id,
            question=case.question.strip().splitlines()[0][:80],
            passed=matched is not None,
            matched_phrase=matched,
            top_titles=titles,
        ))

    passed = sum(1 for r in results if r.passed)
    summary = {
        "total":  len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "recall_at_k": round(passed / len(results), 3) if results else 0.0,
        "k": k,
    }
    return results, summary


def print_report(results: list[CaseResult], summary: dict) -> None:
    print(f"\nRAG eval — recall@{summary['k']} = {summary['recall_at_k']:.1%} "
          f"({summary['passed']}/{summary['total']} cases passed)\n")
    for r in results:
        flag = "PASS" if r.passed else "FAIL"
        print(f"  [{flag}] {r.id}")
        print(f"          {r.question}")
        if r.passed:
            print(f"          matched: '{r.matched_phrase}'")
        else:
            print(f"          top titles: {r.top_titles[:3]}")
    print()


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if len(sys.argv) < 2:
        print("usage: python -m annix_intel.rag.eval <rag_eval.yaml> [k]")
        sys.exit(2)
    cases = load_eval_file(sys.argv[1])
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    results, summary = run_eval(cases, k=k)
    print_report(results, summary)
    # Machine-readable output for CI
    Path("rag_eval_report.json").write_text(
        json.dumps({"summary": summary, "results": [asdict(r) for r in results]},
                   indent=2, default=str),
        encoding="utf-8",
    )
    sys.exit(0 if summary["passed"] == summary["total"] else 1)
