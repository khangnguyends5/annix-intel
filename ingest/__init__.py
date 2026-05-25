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

from annix_intel.ingest.types import WellLog, DepositRecord, GeologicLayer
from annix_intel.ingest.las  import read_las
from annix_intel.ingest.ags  import (
    fetch_ags_wells,
    fetch_ags_formation_tops,
    AGSError,
)
from annix_intel.ingest.smdi import fetch_smdi, SMDIError

__all__ = [
    "WellLog", "DepositRecord", "GeologicLayer",
    "read_las",
    "fetch_ags_wells", "fetch_ags_formation_tops", "AGSError",
    "fetch_smdi", "SMDIError",
]
