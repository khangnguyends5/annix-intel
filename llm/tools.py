"""
Claude tool definitions — the JSON-schemas Claude sees in tool_use mode plus
the Python dispatcher that actually runs them.

Add a new connector:
    1. Write the connector in annix_intel/ingest/<name>.py returning a canonical type.
    2. Add a TOOL_DEFINITIONS entry with the JSON schema Claude should see.
    3. Add a `run_tool` branch that maps it to the function.

That's all — the orchestrator picks it up automatically.
"""

from __future__ import annotations
from typing import Any
import logging

from annix_intel.ingest import (
    read_las,
    fetch_smdi,
    fetch_ags_wells,
    fetch_ags_formation_tops,
)

log = logging.getLogger(__name__)


# ─── TOOL DEFINITIONS (Anthropic Messages API tool_use format) ───────────────
TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "read_las_file",
        "description": (
            "Parse a customer-supplied .las (Log ASCII Standard) well-log file "
            "and return its header (UWI, location, total depth) plus a summary "
            "of curves present (GR, RHOB, NPHI, RT, ...). Use when the user "
            "uploads a .las file or asks about a specific well log."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the .las file.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "fetch_smdi_deposits",
        "description": (
            "Query the Saskatchewan Mineral Deposits Index. Returns a list of "
            "deposit records (name, commodities, status, location, weblink). "
            "Use this for mineral exploration questions in Saskatchewan."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "commodity": {
                    "type": "string",
                    "description": "Commodity name (e.g. Lithium, Uranium, Cobalt). Omit for all.",
                },
                "bbox_utm13": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "[xmin, ymin, xmax, ymax] in EPSG:2151 (UTM 13N).",
                },
                "max_records": {
                    "type": "integer",
                    "description": "Maximum records to return (default 200).",
                    "default": 200,
                },
            },
        },
    },
    {
        "name": "fetch_ags_wells",
        "description": (
            "Query the Alberta Geological Survey for wells with formation picks. "
            "Returns header info + the depth at which each well intersected the "
            "named formation. Use for WCSB hydrogen exploration questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "formation": {
                    "type": "string",
                    "description": "Formation name (e.g. Duperow, Leduc, Beaverhill Lake).",
                },
                "bbox": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "[min_lon, min_lat, max_lon, max_lat] in WGS84.",
                },
                "max_records": {
                    "type": "integer",
                    "default": 200,
                },
            },
        },
    },
    {
        "name": "fetch_ags_formation_layer",
        "description": (
            "Fetch all AGS picks for one formation across a bounding box as a "
            "spatial point layer (each point = one well's pick of that formation). "
            "Use to build structure maps or check depth-to-top for the customer's claim block."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "formation": {"type": "string"},
                "bbox": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "max_records": {"type": "integer", "default": 500},
            },
            "required": ["formation"],
        },
    },
]


# ─── DISPATCHER ──────────────────────────────────────────────────────────────
def run_tool(name: str, args: dict[str, Any]) -> dict:
    """
    Execute a tool call and return a JSON-safe result for Claude.
    Wraps every connector exception so a tool failure doesn't kill the
    conversation — Claude sees the error and can recover.
    """
    log.info("tool_use: %s(%s)", name, args)
    try:
        if name == "read_las_file":
            wl = read_las(args["path"])
            return {"ok": True, "well": wl.summary()}

        if name == "fetch_smdi_deposits":
            records = fetch_smdi(
                commodity=args.get("commodity"),
                bbox_utm13=tuple(args["bbox_utm13"]) if args.get("bbox_utm13") else None,
                max_records=int(args.get("max_records", 200)),
            )
            return {
                "ok": True,
                "count": len(records),
                "deposits": [r.summary() for r in records[:50]],   # truncate for context
                "truncated": len(records) > 50,
            }

        if name == "fetch_ags_wells":
            wells = fetch_ags_wells(
                formation=args.get("formation"),
                bbox=tuple(args["bbox"]) if args.get("bbox") else None,
                max_records=int(args.get("max_records", 200)),
            )
            return {
                "ok": True,
                "count": len(wells),
                "wells": [w.summary() for w in wells[:50]],
                "truncated": len(wells) > 50,
            }

        if name == "fetch_ags_formation_layer":
            layer = fetch_ags_formation_tops(
                formation=args["formation"],
                bbox=tuple(args["bbox"]) if args.get("bbox") else None,
                max_records=int(args.get("max_records", 500)),
            )
            gdf = layer.features
            return {
                "ok": True,
                "name": layer.name,
                "n_picks": int(len(gdf)) if gdf is not None else 0,
                "depth_stats": (
                    {
                        "min": float(gdf["PICK_DEPTH"].min()),
                        "max": float(gdf["PICK_DEPTH"].max()),
                        "mean": float(gdf["PICK_DEPTH"].mean()),
                    } if gdf is not None and "PICK_DEPTH" in gdf.columns and len(gdf) else None
                ),
                "sample_features": (
                    gdf[["UWI", "PICK_DEPTH"]].head(10).to_dict("records")
                    if gdf is not None and len(gdf) else []
                ),
            }

        return {"ok": False, "error": f"Unknown tool: {name}"}

    except Exception as e:                                          # noqa: BLE001
        log.exception("Tool '%s' failed", name)
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
