"""
Canonical types returned by every connector. Keep these stable — they're the
contract between ingestion and everything downstream (scoring, physics, LLM).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd


# ─── WellLog ─────────────────────────────────────────────────────────────────
@dataclass
class WellLog:
    """One well from any source (LAS file, AGS API, customer upload)."""
    uwi:          str                       # Unique Well Identifier
    name:         str | None = None
    operator:     str | None = None
    location:     tuple[float, float] | None = None   # (lon, lat) WGS84
    kb_elev_m:    float | None = None    # Kelly Bushing elevation
    td_m:         float | None = None    # Total Depth
    spud_date:    datetime | None = None
    formation_tops: dict[str, float] = field(default_factory=dict)
    curves:       pd.DataFrame | None = None   # index=DEPT, columns=GR,RHOB,...
    source:       str = "unknown"           # "LAS:filename", "AGS:UWI", "customer-upload"
    raw:          dict[str, Any] | None = field(default=None, repr=False)

    def summary(self) -> dict:
        """JSON-safe summary for Claude / dossiers."""
        return {
            "uwi": self.uwi,
            "name": self.name,
            "operator": self.operator,
            "location": self.location,
            "kb_elev_m": self.kb_elev_m,
            "td_m": self.td_m,
            "spud_date": self.spud_date.isoformat() if self.spud_date else None,
            "formation_tops": self.formation_tops,
            "n_curves": len(self.curves.columns) if self.curves is not None else 0,
            "n_samples": len(self.curves) if self.curves is not None else 0,
            "source": self.source,
        }


# ─── DepositRecord ───────────────────────────────────────────────────────────
@dataclass
class DepositRecord:
    """One point feature from a deposit index (SMDI, MRDS, etc.)."""
    id:                   str
    name:                 str | None = None
    primary_commodities:  list[str] = field(default_factory=list)
    associated_commodities: list[str] = field(default_factory=list)
    status:               str | None = None         # "mine", "deposit", "occurrence", ...
    discovery_type:       str | None = None
    production:           bool = False
    reserves_defined:     bool = False
    location:             tuple[float, float] | None = None   # (x, y) in source CRS
    crs:                  str = "EPSG:4326"
    weblink:              str | None = None
    host_mineral:         str | None = None
    source:               str = "unknown"

    def summary(self) -> dict:
        d = asdict(self)
        return d


# ─── GeologicLayer ───────────────────────────────────────────────────────────
@dataclass
class GeologicLayer:
    """
    A spatial layer: formation top surface, fault trace, gravity anomaly grid.
    Wrapped lightly — heavy spatial operations stay in geopandas/rasterio.
    """
    name:        str
    layer_type:  str                         # "formation_top" | "fault" | "raster"
    description: str | None = None
    crs:         str = "EPSG:4326"
    features:    Any | None = None        # GeoDataFrame or numpy array
    metadata:    dict[str, Any] = field(default_factory=dict)
    source:      str = "unknown"
