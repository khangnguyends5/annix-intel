"""
Canonical types returned by every connector. Keep these stable — they're the
contract between ingestion and everything downstream (scoring, physics, LLM).
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from datetime import datetime
import pandas as pd


# ─── WellLog ─────────────────────────────────────────────────────────────────
@dataclass
class WellLog:
    """One well from any source (LAS file, AGS API, customer upload)."""
    uwi:          str                       # Unique Well Identifier
    name:         Optional[str] = None
    operator:     Optional[str] = None
    location:     Optional[tuple[float, float]] = None   # (lon, lat) WGS84
    kb_elev_m:    Optional[float] = None    # Kelly Bushing elevation
    td_m:         Optional[float] = None    # Total Depth
    spud_date:    Optional[datetime] = None
    formation_tops: dict[str, float] = field(default_factory=dict)
    curves:       Optional[pd.DataFrame] = None   # index=DEPT, columns=GR,RHOB,...
    source:       str = "unknown"           # "LAS:filename", "AGS:UWI", "customer-upload"
    raw:          Optional[dict[str, Any]] = field(default=None, repr=False)

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
    name:                 Optional[str] = None
    primary_commodities:  list[str] = field(default_factory=list)
    associated_commodities: list[str] = field(default_factory=list)
    status:               Optional[str] = None         # "mine", "deposit", "occurrence", ...
    discovery_type:       Optional[str] = None
    production:           bool = False
    reserves_defined:     bool = False
    location:             Optional[tuple[float, float]] = None   # (x, y) in source CRS
    crs:                  str = "EPSG:4326"
    weblink:              Optional[str] = None
    host_mineral:         Optional[str] = None
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
    description: Optional[str] = None
    crs:         str = "EPSG:4326"
    features:    Optional[Any] = None        # GeoDataFrame or numpy array
    metadata:    dict[str, Any] = field(default_factory=dict)
    source:      str = "unknown"
