"""
Saskatchewan Mineral Deposits Index (SMDI) connector — canonical-type wrapper
around the existing `annix_geo_minerals/01_ingestion.py` logic, so both
Annix Geo Minerals and Annix Geo H2 can use the same fetch path through
annix_intel.

The legacy module returns a pandas DataFrame; we normalise into DepositRecord
so downstream code (LLM tools, dossier builder) is data-source agnostic.
"""

from __future__ import annotations

import logging

import requests

from annix_intel.ingest.types import DepositRecord

log = logging.getLogger(__name__)

SMDI_BASE = (
    "https://gis.saskatchewan.ca/egis/rest/services/Economy/"
    "Mineral_Exploration/FeatureServer"
)
LAYER_SMDI = 2
DEFAULT_TIMEOUT_S = 30


class SMDIError(RuntimeError):
    """Raised when the SMDI service returns an error."""


def fetch_smdi(
    commodity: str | None = None,
    bbox_utm13: tuple[float, float, float, float] | None = None,
    max_records: int = 2000,
    out_sr: int = 2151,        # NAD83 / UTM Zone 13N — Sask standard
) -> list[DepositRecord]:
    """
    Pull SMDI Layer 2 (Mineral Deposits Index) and return canonical records.

    Parameters
    ----------
    commodity
        Substring match on PRIMARYCOMMODITIES (e.g. "Lithium", "Uranium").
        None = all commodities.
    bbox_utm13
        (xmin, ymin, xmax, ymax) in EPSG:2151 to restrict the query.
    max_records
        ArcGIS REST page size.
    out_sr
        Output spatial reference. Default 2151 (UTM13N).

    Returns
    -------
    list[DepositRecord]
    """
    where = (f"PRIMARYCOMMODITIES LIKE '%{commodity}%'"
             if commodity else "1=1")

    params: dict = {
        "where":             where,
        "outFields":         "*",
        "outSR":             out_sr,
        "f":                 "json",
        "resultRecordCount": max_records,
    }
    if bbox_utm13:
        params["geometry"] = ",".join(map(str, bbox_utm13))
        params["geometryType"] = "esriGeometryEnvelope"
        params["inSR"] = out_sr
        params["spatialRel"] = "esriSpatialRelIntersects"

    url = f"{SMDI_BASE}/{LAYER_SMDI}/query"
    log.info("SMDI fetch: commodity=%s bbox=%s", commodity, bbox_utm13)

    try:
        r = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT_S)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        raise SMDIError(f"SMDI request failed: {e}") from e

    if "error" in data:
        raise SMDIError(f"SMDI error response: {data['error']}")

    out: list[DepositRecord] = []
    for feat in data.get("features", []):
        a = feat.get("attributes", {})
        g = feat.get("geometry", {})
        out.append(DepositRecord(
            id=str(a.get("SMDI") or a.get("OBJECTID")),
            name=a.get("NAME"),
            primary_commodities=_split_csv(a.get("PRIMARYCOMMODITIES")),
            associated_commodities=_split_csv(a.get("ASSOCIATEDCOMMODITIES")),
            status=a.get("SYMBOLOGY_STATUS"),
            discovery_type=a.get("DISCOVERYTYPE"),
            production=_to_bool(a.get("PRODUCTION")),
            reserves_defined=_to_bool(a.get("RESERVESRESOURCES")),
            location=(g.get("x"), g.get("y")) if g else None,
            crs=f"EPSG:{out_sr}",
            weblink=a.get("WEBLINK"),
            host_mineral=a.get("HOSTMINERAL"),
            source="SMDI",
        ))

    log.info("SMDI fetch: %d records", len(out))
    return out


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _split_csv(v) -> list[str]:
    if not v:
        return []
    return [x.strip() for x in str(v).split(",") if x.strip()]


def _to_bool(v) -> bool:
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("yes", "true", "1", "y")


# ─── CLI / smoke ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    commodity = sys.argv[1] if len(sys.argv) > 1 else "Lithium"
    records = fetch_smdi(commodity=commodity, max_records=50)
    print(f"Found {len(records)} {commodity} deposits.")
    for r in records[:3]:
        print(json.dumps(r.summary(), indent=2, default=str))
