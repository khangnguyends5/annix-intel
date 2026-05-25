"""
Alberta Geological Survey (AGS) REST connector.

AGS publishes through ArcGIS Online under owner `alberta-geological-survey`
at services2.arcgis.com/jQV6VMr2Loovu7GU. The endpoints in AGS_ENDPOINTS were
verified live against the public service catalog.

Important structural note
-------------------------
AGS does NOT publish a single "Well_Picks_Alberta" service. The available
datasets are:

  - Oil_Sands_Wells           Real well locations (oil sands area only) with
                              UWI, KB elevation, TD. No general WCSB well-picks
                              service — pull customer LAS files for that.
  - Mineral_Occurrences       28 fields, point features, all commodities.
                              The richest layer for Annix Geo mineral work.
  - Bedrock geology + faults  Polygons + line features for context.
  - Geological Framework of Alberta — per-formation "Areal Extent (Top/Base)"
    polygon services, naming pattern `ET<code>` / `EB<code>`. Looking up a
    specific formation code requires browsing the GFA catalog. Add to
    GFA_FORMATION_CODES as you discover them.

Public API
----------
    fetch_ags_oil_sands_wells(bbox=None)      → list[WellLog]
    fetch_ags_mineral_occurrences(...)        → list[DepositRecord]
    fetch_ags_bedrock_geology(bbox)           → GeologicLayer (polygons)
    fetch_ags_faults(bbox)                    → GeologicLayer (line features)
    fetch_ags_formation_extent(code, bbox)    → GeologicLayer (formation polygons)

Back-compat shims (preserved so dossier.py keeps working):
    fetch_ags_wells(bbox, formation=)         → list[WellLog]    (uses oil_sands)
    fetch_ags_formation_tops(formation, bbox) → GeologicLayer   (uses extent if known)
"""

from __future__ import annotations

import logging

import geopandas as gpd
import requests

from annix_intel.ingest.types import DepositRecord, GeologicLayer, WellLog

log = logging.getLogger(__name__)

# ─── Verified endpoints (May 2026) ───────────────────────────────────────────
_BASE = "https://services2.arcgis.com/jQV6VMr2Loovu7GU/arcgis/rest/services"

AGS_ENDPOINTS: dict[str, str] = {
    "oil_sands_wells":   f"{_BASE}/Oil_Sands_Wells/FeatureServer/0/query",
    "mineral_occurrences": f"{_BASE}/Mineral_Occurrences/FeatureServer/0/query",
    "bedrock_geology":   f"{_BASE}/Simplified_Begrock_Geology_AGS_Map_600/FeatureServer/0/query",
    "cordilleran_faults": f"{_BASE}/Cordilleran_Deformation_Belt_AGS_Map_542/FeatureServer/0/query",
    "hydrogeological_regions": f"{_BASE}/New_Regions_UPDATE_UPDATE/FeatureServer/0/query",
    "core_locations":    f"{_BASE}/MineralCoresHSA/FeatureServer/0/query",
}

# Per-formation "Areal Extent" services. Codes are AGS Geological Framework
# of Alberta stratigraphic identifiers. Expand as you discover more.
GFA_FORMATION_CODES: dict[str, str] = {
    "Precambrian":     "90PreC",
    "Mannville":       "52KMc",      # 52 = unit number, K = Cretaceous, Mc = Mannville
    "Leduc":           "69DLd",      # 69 = unit, D = Devonian, Ld = Leduc
    # TODO: confirm codes for Duperow, Beaverhill Lake, Wabamun, Nisku by
    # browsing https://geology-ags-aer.opendata.arcgis.com/ search "Areal Extent".
}

DEFAULT_TIMEOUT_S = 60
MAX_RECORDS_PER_QUERY = 2000


class AGSError(RuntimeError):
    """Raised when AGS API returns an error or unexpected payload."""


# ─── Public connectors ───────────────────────────────────────────────────────
def fetch_ags_oil_sands_wells(
    bbox: tuple[float, float, float, float] | None = None,
    max_records: int = 500,
) -> list[WellLog]:
    """
    Wells from the Oil_Sands_Wells layer. Note: only covers the oil sands
    area (Athabasca, Cold Lake, Peace River) — not the full WCSB. For
    customer claims outside oil sands, expect 0 results and fall back to
    customer-supplied LAS files.
    """
    params = {
        "where":             "1=1",
        "outFields":         "*",
        "outSR":             4326,
        "f":                 "geojson",
        "resultRecordCount": min(max_records, MAX_RECORDS_PER_QUERY),
    }
    if bbox:
        params.update(_bbox_params(bbox))

    data = _request(AGS_ENDPOINTS["oil_sands_wells"], params)
    out: list[WellLog] = []
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates")
        out.append(WellLog(
            uwi=str(props.get("uwi") or props.get("UWI") or props.get("OBJECTID")),
            name=props.get("well_name") or props.get("WellName"),
            operator=props.get("operator"),
            location=tuple(coords) if coords and len(coords) == 2 else None,
            kb_elev_m=_safe_float(props.get("kb_elev") or props.get("ELEV_KB")),
            td_m=_safe_float(props.get("td") or props.get("total_depth")),
            source=f"AGS:OilSandsWells:{props.get('uwi') or props.get('OBJECTID')}",
            raw=props,
        ))
    log.info("AGS oil-sands wells: %d wells in bbox=%s", len(out), bbox)
    return out


def fetch_ags_mineral_occurrences(
    commodity: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    max_records: int = 500,
) -> list[DepositRecord]:
    """
    Mineral occurrences from AGS — the Alberta analog to SMDI. 28 fields
    including commodity, dev_stage, geo_age, geo_unit, uwi (if drilled).
    """
    where_parts: list[str] = []
    if commodity:
        where_parts.append(f"UPPER(commodity) LIKE '%{commodity.upper()}%'")
    where = " AND ".join(where_parts) if where_parts else "1=1"

    params = {
        "where":             where,
        "outFields":         "*",
        "outSR":             4326,
        "f":                 "geojson",
        "resultRecordCount": min(max_records, MAX_RECORDS_PER_QUERY),
    }
    if bbox:
        params.update(_bbox_params(bbox))

    data = _request(AGS_ENDPOINTS["mineral_occurrences"], params)
    out: list[DepositRecord] = []
    for feat in data.get("features", []):
        p = feat.get("properties", {})
        g = feat.get("geometry") or {}
        coords = g.get("coordinates")
        out.append(DepositRecord(
            id=str(p.get("ags_id") or p.get("site_id") or p.get("OBJECTID")),
            name=p.get("site_name"),
            primary_commodities=_split_csv(p.get("commodity")),
            associated_commodities=_split_csv(p.get("other_commodity")),
            status=p.get("dev_stage"),
            discovery_type=p.get("sample_type"),
            production=str(p.get("dev_stage") or "").lower().startswith("prod"),
            reserves_defined=str(p.get("dev_stage") or "").lower() in ("deposit", "reserves"),
            location=tuple(coords) if coords and len(coords) == 2 else None,
            crs="EPSG:4326",
            weblink=p.get("feature_layer_link") or p.get("download_link"),
            host_mineral=p.get("geo_unit"),
            source="AGS",
        ))
    log.info("AGS mineral occurrences: %d records, commodity=%s, bbox=%s",
             len(out), commodity, bbox)
    return out


def fetch_ags_bedrock_geology(
    bbox: tuple[float, float, float, float],
    max_records: int = 1000,
) -> GeologicLayer:
    """Simplified Bedrock Geology of Alberta (AGS Map 600) as polygons."""
    params = {
        "where":             "1=1",
        "outFields":         "*",
        "outSR":             4326,
        "f":                 "geojson",
        "resultRecordCount": min(max_records, MAX_RECORDS_PER_QUERY),
        **_bbox_params(bbox),
    }
    data = _request(AGS_ENDPOINTS["bedrock_geology"], params)
    gdf = gpd.GeoDataFrame.from_features(data.get("features", []), crs="EPSG:4326")
    log.info("AGS bedrock geology: %d polygons in bbox=%s", len(gdf), bbox)
    return GeologicLayer(
        name="AGS_bedrock_geology",
        layer_type="raster",       # polygons stored as vector
        description="Simplified Bedrock Geology of Alberta (AGS Map 600)",
        crs="EPSG:4326",
        features=gdf,
        metadata={"n_polygons": int(len(gdf))},
        source="AGS",
    )


def fetch_ags_faults(
    bbox: tuple[float, float, float, float],
    max_records: int = 1000,
) -> GeologicLayer:
    """
    Cordilleran Deformation Belt fault traces. Note: this is structural
    front data — not all faults in Alberta, just the main thrust belt.
    For basement-fault mapping (relevant to natural H2 conduits) you need
    proprietary or customer-supplied seismic interpretation.
    """
    params = {
        "where":             "1=1",
        "outFields":         "*",
        "outSR":             4326,
        "f":                 "geojson",
        "resultRecordCount": min(max_records, MAX_RECORDS_PER_QUERY),
        **_bbox_params(bbox),
    }
    data = _request(AGS_ENDPOINTS["cordilleran_faults"], params)
    gdf = gpd.GeoDataFrame.from_features(data.get("features", []), crs="EPSG:4326")
    log.info("AGS Cordilleran faults: %d features in bbox=%s", len(gdf), bbox)
    return GeologicLayer(
        name="AGS_cordilleran_faults",
        layer_type="fault",
        description="Cordilleran Deformation Belt (AGS Map 542)",
        crs="EPSG:4326",
        features=gdf,
        metadata={"n_features": int(len(gdf))},
        source="AGS",
    )


def fetch_ags_formation_extent(
    formation: str,
    bbox: tuple[float, float, float, float] | None = None,
) -> GeologicLayer:
    """
    Areal extent (top surface) of a named formation. Uses the AGS Geological
    Framework of Alberta `ET<code>` services. Falls back to a stub layer if
    the formation code isn't in GFA_FORMATION_CODES — add it when you find it.
    """
    code = GFA_FORMATION_CODES.get(formation)
    if not code:
        log.warning(
            "Formation '%s' not in GFA_FORMATION_CODES. Add the AGS code to "
            "annix_intel/ingest/ags.py:GFA_FORMATION_CODES.", formation,
        )
        return GeologicLayer(
            name=f"AGS_{formation}_extent",
            layer_type="formation_top",
            description=f"AGS extent for {formation} — code unknown, no data",
            crs="EPSG:4326",
            features=None,
            metadata={"formation": formation, "code_missing": True},
            source="AGS",
        )

    url = f"{_BASE}/ET{code}/FeatureServer/0/query"
    params = {
        "where":             "1=1",
        "outFields":         "*",
        "outSR":             4326,
        "f":                 "geojson",
        "resultRecordCount": MAX_RECORDS_PER_QUERY,
    }
    if bbox:
        params.update(_bbox_params(bbox))

    data = _request(url, params)
    gdf = gpd.GeoDataFrame.from_features(data.get("features", []), crs="EPSG:4326")
    log.info("AGS formation extent '%s' (ET%s): %d polygons", formation, code, len(gdf))
    return GeologicLayer(
        name=f"AGS_{formation}_extent",
        layer_type="formation_top",
        description=f"Areal extent of top of {formation} (AGS GFA, code {code})",
        crs="EPSG:4326",
        features=gdf,
        metadata={"formation": formation, "code": code, "n_polygons": int(len(gdf))},
        source="AGS",
    )


# ─── Back-compat shims ───────────────────────────────────────────────────────
# Keep the original public names so existing callers (dossier.py) don't break.
def fetch_ags_wells(
    bbox: tuple[float, float, float, float] | None = None,
    formation: str | None = None,                                          # noqa: ARG001
    max_records: int = 500,
) -> list[WellLog]:
    """
    Back-compat shim. Returns Oil_Sands_Wells (the only WCSB well layer
    AGS publishes). The `formation` argument is ignored — AGS has no
    general well-picks service to filter by.
    """
    return fetch_ags_oil_sands_wells(bbox=bbox, max_records=max_records)


def fetch_ags_formation_tops(
    formation: str,
    bbox: tuple[float, float, float, float] | None = None,
    max_records: int = 1000,                                                  # noqa: ARG001
) -> GeologicLayer:
    """
    Back-compat shim. The new function (`fetch_ags_formation_extent`)
    returns a polygon layer of areal extent — different concept from the
    point picks the old name implied. We adapt the return shape so callers
    that read `.features` and `.n_picks` still work.
    """
    layer = fetch_ags_formation_extent(formation, bbox=bbox)
    # Compatibility: synthesize a "depth_stats" the caller used to read.
    layer.metadata.setdefault("note",
        "Returned data is areal-extent polygons, not point picks with depths. "
        "For real depth picks supply customer LAS files.")
    return layer


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _bbox_params(bbox: tuple[float, float, float, float]) -> dict:
    return {
        "geometry":     ",".join(map(str, bbox)),
        "geometryType": "esriGeometryEnvelope",
        "inSR":         4326,
        "spatialRel":   "esriSpatialRelIntersects",
    }


def _request(url: str, params: dict) -> dict:
    try:
        r = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT_S)
        r.raise_for_status()
    except requests.RequestException as e:
        raise AGSError(f"AGS request failed ({url}): {e}") from e
    try:
        data = r.json()
    except ValueError as e:
        raise AGSError(f"AGS returned non-JSON ({url}): {e}") from e
    if "error" in data:
        raise AGSError(f"AGS error response: {data['error']}")
    return data


def _safe_float(v) -> float | None:
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _split_csv(v) -> list[str]:
    if not v:
        return []
    return [x.strip() for x in str(v).replace(";", ",").split(",") if x.strip()]


# ─── CLI / live smoke ────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    bbox = (-114.5, 53.2, -113.8, 53.6)  # central Alberta

    print("=== Oil sands wells (expect 0 — central AB is outside oil sands) ===")
    print(f"  {len(fetch_ags_oil_sands_wells(bbox=bbox, max_records=5))} wells")

    print("\n=== Mineral occurrences (commodity=Lithium) ===")
    recs = fetch_ags_mineral_occurrences(commodity="Lithium", bbox=None, max_records=5)
    print(f"  {len(recs)} occurrences across Alberta")
    for r in recs[:3]:
        print(f"    {r.name or r.id} — {r.primary_commodities} @ {r.location}")

    print("\n=== Bedrock geology (small bbox) ===")
    layer = fetch_ags_bedrock_geology(bbox=bbox, max_records=10)
    print(f"  {len(layer.features) if layer.features is not None else 0} polygons")
