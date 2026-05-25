"""
annix_intel.ingest — Data ingestion connectors.

Canonical types
---------------
Every connector returns one of:

    WellLog          One well: header (location, depth range) + curves (DataFrame).
    DepositRecord    Point feature: location, commodities, status, source URL.
    GeologicLayer    Vector / raster overlay: formation tops, faults, etc.

These are intentionally minimal — the brain (annix_intel.llm) can compose them
into bigger structures. Adding a new data source = writing one connector that
emits canonical types. Nothing else has to change.

Public connectors
-----------------
    read_las(path)                       → WellLog
    fetch_smdi(...)                      → list[DepositRecord]
    fetch_ags_wells(...)                 → list[WellLog]            (header-only by default)
    fetch_ags_formation_tops(area_wkt)   → list[GeologicLayer]

Each connector has a corresponding Claude tool definition in annix_intel.llm.tools.
"""

from annix_intel.ingest.ags import (
    AGSError,
    fetch_ags_bedrock_geology,
    fetch_ags_faults,
    fetch_ags_formation_extent,
    fetch_ags_formation_tops,
    fetch_ags_mineral_occurrences,
    # Canonical AGS connectors
    fetch_ags_oil_sands_wells,
    # Back-compat shims (preserved for dossier.py)
    fetch_ags_wells,
)
from annix_intel.ingest.las import read_las
from annix_intel.ingest.segy import read_segy
from annix_intel.ingest.segy import summarise_for_llm as segy_summary
from annix_intel.ingest.smdi import SMDIError, fetch_smdi
from annix_intel.ingest.types import DepositRecord, GeologicLayer, WellLog

__all__ = [
    "WellLog", "DepositRecord", "GeologicLayer",
    "read_las",
    "read_segy", "segy_summary",
    "fetch_ags_oil_sands_wells", "fetch_ags_mineral_occurrences",
    "fetch_ags_bedrock_geology", "fetch_ags_faults", "fetch_ags_formation_extent",
    "fetch_ags_wells", "fetch_ags_formation_tops",
    "AGSError",
    "fetch_smdi", "SMDIError",
]
