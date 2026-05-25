"""
SEG-Y seismic reader. SEG-Y is the standard format for seismic survey volumes.
Wraps `segyio`. We DELIBERATELY do not load the full trace data into memory —
real survey volumes are 100 GB+. Instead we read:

  - File header (binary header, text header)
  - Survey geometry (inline/crossline ranges, sample count, sample interval)
  - Trace count + first-trace summary

That's enough for Claude to reason about the survey ("you have a 1500-inline
3D over the Wabamun-N block, 4-ms sampling, 6-second record length, OK to
target picks at the Beaverhill horizon") without sucking gigabytes into RAM.

When a customer needs actual interpretation we hand off to OpendTect / Petrel
through a paid integration. This module's job is "Claude can talk about the
file the customer dropped on us".
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from annix_intel.ingest.types import GeologicLayer

log = logging.getLogger(__name__)


def read_segy(
    path: str | Path,
    deep_scan: bool = False,
) -> GeologicLayer:
    """
    Parse a SEG-Y file and return a GeologicLayer summarising the survey.

    Parameters
    ----------
    path
        Path to .segy / .sgy file.
    deep_scan
        If True, scan every trace header to compute inline/crossline ranges
        and bounding box. Slower (~30s for a 5 GB volume) but accurate.
        Default False reads only file header + 1st/last trace — fast (<1s)
        and usually correct for well-formed SEG-Y.

    Returns
    -------
    GeologicLayer with metadata populated:

      sample_interval_ms      Sample interval (e.g. 4.0)
      n_samples_per_trace     Trace length in samples
      record_length_ms        n_samples * sample_interval
      n_traces                Total trace count
      inline_range            (min, max) or None
      crossline_range         (min, max) or None
      bbox_utm                (xmin, ymin, xmax, ymax) in source CDP coords
      crs                     "unknown — check trace coordinate units"
      text_header             First EBCDIC header (40 lines * 80 chars) as text

    Raises
    ------
    FileNotFoundError, RuntimeError if segyio missing or file unreadable.
    """
    try:
        import segyio
    except ImportError as e:                                                  # pragma: no cover
        raise RuntimeError(
            "annix_intel.ingest.segy requires `segyio`. "
            "Install: pip install segyio"
        ) from e

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SEG-Y file not found: {path}")

    meta: dict[str, Any] = {"path": str(path), "size_bytes": path.stat().st_size}

    with segyio.open(str(path), ignore_geometry=not deep_scan) as f:
        # Text header — usually contains processing history, survey ID.
        try:
            text = segyio.tools.wrap(f.text[0]).strip()
        except Exception:                                                     # noqa: BLE001
            text = ""
        meta["text_header"] = text

        # Binary header — sample rate, n_samples.
        bin_h = f.bin
        meta["sample_interval_us"] = int(bin_h[segyio.BinField.Interval])
        meta["sample_interval_ms"] = meta["sample_interval_us"] / 1000.0
        meta["n_samples_per_trace"] = int(bin_h[segyio.BinField.Samples])
        meta["record_length_ms"] = round(
            meta["n_samples_per_trace"] * meta["sample_interval_ms"], 1,
        )
        meta["format_code"] = int(bin_h[segyio.BinField.Format])
        meta["n_traces"] = int(f.tracecount)

        if deep_scan:
            # Try the structured geometry — only works for regular grids.
            try:
                meta["inline_range"]    = (int(f.ilines.min()), int(f.ilines.max()))
                meta["crossline_range"] = (int(f.xlines.min()), int(f.xlines.max()))
                meta["geometry"] = "structured-3D"
            except Exception:                                                 # noqa: BLE001
                meta["geometry"] = "unstructured-or-2D"

            # CDP X/Y bbox from trace headers — these usually carry UTM coords
            # with a scaling factor in CoordinateUnits / SourceGroupScalar.
            try:
                f.mmap()
                xs = f.attributes(segyio.TraceField.CDP_X)[:]
                ys = f.attributes(segyio.TraceField.CDP_Y)[:]
                scalar = int(f.header[0][segyio.TraceField.SourceGroupScalar] or 1)
                # SEG-Y scalar: negative means divide, positive means multiply.
                if scalar < 0:
                    xs = xs / abs(scalar)
                    ys = ys / abs(scalar)
                elif scalar > 1:
                    xs = xs * scalar
                    ys = ys * scalar
                meta["bbox_source_coords"] = (
                    float(xs.min()), float(ys.min()),
                    float(xs.max()), float(ys.max()),
                )
                meta["coord_scalar"] = scalar
            except Exception as e:                                            # noqa: BLE001
                log.debug("SEG-Y bbox extraction failed: %s", e)
                meta["bbox_source_coords"] = None
        else:
            meta["geometry"] = "header-only-scan"

    name = path.stem
    return GeologicLayer(
        name=f"SEGY_{name}",
        layer_type="raster",
        description=(
            f"Seismic survey ({meta['n_traces']} traces, "
            f"{meta['n_samples_per_trace']} samples @ {meta['sample_interval_ms']} ms, "
            f"{meta['record_length_ms']} ms record)"
        ),
        crs="unknown",
        features=None,                       # don't ship trace data through canonical type
        metadata=meta,
        source=f"SEGY:{path.name}",
    )


def summarise_for_llm(layer: GeologicLayer, header_lines: int = 8) -> dict:
    """Compact, JSON-safe dict for stuffing into a Claude tool result."""
    m = layer.metadata
    text = (m.get("text_header") or "").splitlines()
    return {
        "filename":          Path(m["path"]).name,
        "size_mb":           round(m["size_bytes"] / 1e6, 1),
        "n_traces":          m["n_traces"],
        "samples_per_trace": m["n_samples_per_trace"],
        "sample_interval_ms": m["sample_interval_ms"],
        "record_length_ms":  m["record_length_ms"],
        "inline_range":      m.get("inline_range"),
        "crossline_range":   m.get("crossline_range"),
        "bbox_source_coords": m.get("bbox_source_coords"),
        "geometry":          m["geometry"],
        "text_header_preview": "\n".join(text[:header_lines]),
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m annix_intel.ingest.segy <file.sgy> [--deep]")
        sys.exit(2)
    deep = "--deep" in sys.argv
    layer = read_segy(sys.argv[1], deep_scan=deep)
    print(json.dumps(summarise_for_llm(layer), indent=2, default=str))
