"""
Alberta Geological Survey (AGS) REST connector.

AGS exposes a public ArcGIS FeatureServer at:
    https://geology-ags-aer.opendata.arcgis.com/

For programmatic access we use the underlying ArcGIS REST endpoints.
This module wraps two of the most useful for hydrogen/mineral exploration:

    1. Well bore picks / formation tops  → list[WellLog] (header-only)
    2. Bedrock geology / formation traces  → list[GeologicLayer]

Production-grade extensions to add later
----------------------------------------
- Aquifer chemistry observations (AEP groundwater monitoring network)
- Surface geochemistry surveys
- 3D structural model (Geological Framework of Alberta)
- Drilling logs for AER-licensed wells (requires AER credentials)
"""

from __future__ import annotations
from typing import Optional
import logging
import requests
import geopandas as gpd

from annix_intel.ingest.types import WellLog, GeologicLayer

log = logging.getLogger(__name__)

# ─── ENDPOINTS ───────────────────────────────────────────────────────────────
# These are the public AGS Open Data endpoints. They are stable but if AGS
# republishes them under a different service name, only this dict needs updating.
AGS_ENDPOINTS = {
    "well_picks": (
        "https://services.arcgis.com/atVRTeq7n4erfaEC/arcgis/rest/services/"
        "Well_Picks_Alberta/FeatureServer/0/query"
    ),
    "bedrock_geology": (
        "https://services.arcgis.com/atVRTeq7n4erfaEC/arcgis/rest/services/"
        "Bedrock_Geology_of_Alberta/FeatureServer/0/query"
    ),
}

DEFAULT_TIMEOUT_S = 60


class AGSError(RuntimeError):
    """Raised when AGS API returns an error or unexpected payload."""


# ─── Wells (header-only via picks) ───────────────────────────────────────────
def fetch_ags_wells(
    bbox: Optional[tuple[float, float, float, float]] = None,
    formation: Optional[str] = None,
    max_records: int = 1000,
) -> list[WellLog]:
    """
    Fetch wells from the AGS formation-picks service. Returns header-only
    WellLog entries (no curves — AGS doesn't serve raw LAS through this API).

    Parameters
    ----------
    bbox
        (min_lon, min_lat, max_lon, max_lat) in WGS84. None = whole province.
    formation
        Filter to wells that have a pick for this formation (e.g. "Duperow").
    max_records
        ArcGIS REST page size; max 2000.

    Returns
    -------
    list[WellLog], one entry per unique UWI in the response. formation_tops
    is populated from the picks.
    """
    where_parts = []
    if formation:
        where_parts.append(f"FORMATION LIKE '%{formation}%'")
    where = " AND ".join(where_parts) if where_parts else "1=1"

    params = {
        "where":             where,
        "outFields":         "*",
        "outSR":             4326,
        "f":                 "geojson",
        "resultRecordCount": min(max_records, 2000),
    }
    if bbox:
        params["geometry"] = ",".join(map(str, bbox))
        params["geometryType"] = "esriGeometryEnvelope"
        params["inSR"] = 4326
        params["spatialRel"] = "esriSpatialRelIntersects"

    log.info("AGS well picks: where=%s bbox=%s", where, bbox)
    data = _request(AGS_ENDPOINTS["well_picks"], params)

    # Group features by UWI; each UWI gets multiple picks (one per formation).
    by_uwi: dict[str, WellLog] = {}
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        uwi   = str(props.get("UWI") or props.get("WELL_ID") or props.get("OBJECTID"))
        geom  = feat.get("geometry") or {}
        coords = geom.get("coordinates")
        location = tuple(coords) if coords and len(coords) == 2 else None

        log_obj = by_uwi.get(uwi)
        if log_obj is None:
            log_obj = WellLog(
                uwi=uwi,
                name=props.get("WELL_NAME"),
                operator=props.get("OPERATOR"),
                location=location,
                kb_elev_m=_safe_float(props.get("KB_ELEV") or props.get("ELEV_KB")),
                td_m=_safe_float(props.get("TD") or props.get("TOTAL_DEPTH")),
                source=f"AGS:{uwi}",
                raw={"first_pick_props": props},
            )
            by_uwi[uwi] = log_obj

        formation_name = props.get("FORMATION") or props.get("STRAT_UNIT")
        depth = _safe_float(props.get("PICK_DEPTH") or props.get("TOP_DEPTH"))
        if formation_name and depth is not None:
            log_obj.formation_tops[formation_name.upper()] = depth

    log.info("AGS well picks: %d unique wells, %d picks total",
             len(by_uwi), len(data.get("features", [])))
    return list(by_uwi.values())


# ─── Formation tops as a spatial layer ───────────────────────────────────────
def fetch_ags_formation_tops(
    formation: str,
    bbox: Optional[tuple[float, float, float, float]] = None,
    max_records: int = 2000,
) -> GeologicLayer:
    """
    Get all picks for one formation as a spatial layer. Useful for building
    structure maps (depth-to-top contours, isopachs, etc.).

    Returns a GeologicLayer wrapping a GeoDataFrame of points with PICK_DEPTH.
    """
    params = {
        "where":             f"FORMATION LIKE '%{formation}%'",
        "outFields":         "UWI,FORMATION,PICK_DEPTH,TOP_DEPTH",
        "outSR":             4326,
        "f":                 "geojson",
        "resultRecordCount": min(max_records, 2000),
    }
    if bbox:
        params["geometry"] = ",".join(map(str, bbox))
        params["geometryType"] = "esriGeometryEnvelope"
        params["inSR"] = 4326

    data = _request(AGS_ENDPOINTS["well_picks"], params)
    gdf = gpd.GeoDataFrame.from_features(data.get("features", []), crs="EPSG:4326")
    log.info("AGS formation '%s': %d picks", formation, len(gdf))
    return GeologicLayer(
        name=f"AGS_{formation}_picks",
        layer_type="formation_top",
        description=f"AGS well picks for {formation}",
        crs="EPSG:4326",
        features=gdf,
        metadata={"formation": formation, "n_picks": len(gdf)},
        source="AGS",
    )


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _request(url: str, params: dict) -> dict:
    try:
        r = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT_S)
        r.raise_for_status()
    except requests.RequestException as e:
        raise AGSError(f"AGS request failed: {e}") from e

    try:
        data = r.json()
    except ValueError as e:
        raise AGSError(f"AGS returned non-JSON payload: {e}") from e

    if "error" in data:
        raise AGSError(f"AGS error response: {data['error']}")
    return data


def _safe_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ─── CLI / smoke ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, json
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    formation = sys.argv[1] if len(sys.argv) > 1 else "Duperow"
    wells = fetch_ags_wells(formation=formation, max_records=20)
    print(f"Found {len(wells)} wells with picks for '{formation}':")
    for w in wells[:5]:
        print(json.dumps(w.summary(), indent=2, default=str))
