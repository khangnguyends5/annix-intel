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

import logging
from typing import Any

from annix_intel.ingest import (
    fetch_ags_bedrock_geology,
    fetch_ags_faults,
    fetch_ags_formation_tops,
    fetch_ags_mineral_occurrences,
    fetch_ags_wells,
    fetch_smdi,
    read_las,
    read_segy,
    segy_summary,
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
            "Fetch AGS Geological Framework of Alberta polygons for the named "
            "formation's areal extent (top surface). Returns coverage/extent, "
            "not point picks. Use to check whether a target formation exists "
            "laterally beneath the customer's claim block."
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
    {
        "name": "fetch_ags_mineral_occurrences",
        "description": (
            "Query Alberta Geological Survey's Mineral Occurrences layer "
            "(the AB analog to Saskatchewan's SMDI). 28 fields including "
            "commodity, development stage, geological age/unit, and UWI for "
            "drilled occurrences. Use for mineral exploration questions in Alberta."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "commodity": {
                    "type": "string",
                    "description": "Commodity name (Lithium, Uranium, Cobalt, Helium, Hydrogen, ...). Omit for all.",
                },
                "bbox": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "[min_lon, min_lat, max_lon, max_lat] WGS84.",
                },
                "max_records": {"type": "integer", "default": 200},
            },
        },
    },
    {
        "name": "fetch_ags_bedrock_geology",
        "description": (
            "Pull AGS Simplified Bedrock Geology (Map 600) polygons for the "
            "given bbox. Use to characterise the geological context — host "
            "rock, ages, contacts — beneath a claim block."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "bbox": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "max_records": {"type": "integer", "default": 500},
            },
            "required": ["bbox"],
        },
    },
    {
        "name": "fetch_ags_faults",
        "description": (
            "Fault traces from AGS Cordilleran Deformation Belt (Map 542). "
            "Note: this only covers the main thrust belt in western AB — not "
            "all basement faults. Use to flag structural fluid pathways "
            "intersecting a claim block."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "bbox": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "max_records": {"type": "integer", "default": 500},
            },
            "required": ["bbox"],
        },
    },
    {
        "name": "read_segy_file",
        "description": (
            "Inspect a customer-supplied SEG-Y seismic file. Returns header + "
            "survey geometry summary (trace count, sample interval, record "
            "length, inline/crossline range, CDP bbox). Does NOT load trace "
            "data — for actual interpretation hand off to OpendTect/Petrel. "
            "Use when the user uploads a .segy/.sgy file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the SEG-Y file.",
                },
                "deep_scan": {
                    "type": "boolean",
                    "description": (
                        "If true, scan all trace headers (slower, ~30s for 5GB) "
                        "to compute accurate geometry. False (default) reads "
                        "only the file header — fast and usually sufficient."
                    ),
                    "default": False,
                },
            },
            "required": ["path"],
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

        if name == "fetch_ags_mineral_occurrences":
            records = fetch_ags_mineral_occurrences(
                commodity=args.get("commodity"),
                bbox=tuple(args["bbox"]) if args.get("bbox") else None,
                max_records=int(args.get("max_records", 200)),
            )
            return {
                "ok": True,
                "count": len(records),
                "occurrences": [r.summary() for r in records[:50]],
                "truncated": len(records) > 50,
            }

        if name == "fetch_ags_bedrock_geology":
            layer = fetch_ags_bedrock_geology(
                bbox=tuple(args["bbox"]),
                max_records=int(args.get("max_records", 500)),
            )
            gdf = layer.features
            return {
                "ok": True,
                "n_polygons": int(len(gdf)) if gdf is not None else 0,
                "fields": list(gdf.columns) if gdf is not None else [],
                "sample": (gdf.drop(columns="geometry").head(5).to_dict("records")
                           if gdf is not None and len(gdf) else []),
            }

        if name == "read_segy_file":
            layer = read_segy(args["path"], deep_scan=bool(args.get("deep_scan", False)))
            return {"ok": True, "segy": segy_summary(layer)}

        if name == "fetch_ags_faults":
            layer = fetch_ags_faults(
                bbox=tuple(args["bbox"]),
                max_records=int(args.get("max_records", 500)),
            )
            gdf = layer.features
            return {
                "ok": True,
                "n_features": int(len(gdf)) if gdf is not None else 0,
                "sample": (gdf.drop(columns="geometry").head(5).to_dict("records")
                           if gdf is not None and len(gdf) else []),
            }

        return {"ok": False, "error": f"Unknown tool: {name}"}

    except Exception as e:                                          # noqa: BLE001
        log.exception("Tool '%s' failed", name)
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
