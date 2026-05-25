"""
LAS file reader. LAS = Log ASCII Standard — every well log on Earth comes in
this format. Wraps `lasio` (the de-facto standard) and normalises into WellLog.

Curve aliases
-------------
LAS files name the same physical measurement many different ways. We normalise
common ones so downstream code can rely on canonical names (GR, RHOB, NPHI, ...).
Customers can override the mapping per project if their dataset uses oddball names.
"""

from __future__ import annotations

from pathlib import Path

try:
    import lasio
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "annix_intel.ingest.las requires `lasio`. Install: pip install lasio"
    ) from e

from annix_intel.ingest.types import WellLog

# Common curve aliases → canonical name. Extend as needed.
CURVE_ALIASES: dict[str, str] = {
    # Gamma ray
    "GR": "GR", "GRD": "GR", "SGR": "GR", "CGR": "GR",
    # Bulk density
    "RHOB": "RHOB", "RHOZ": "RHOB", "DEN": "RHOB", "ZDEN": "RHOB",
    # Neutron porosity
    "NPHI": "NPHI", "TNPH": "NPHI", "NPOR": "NPHI", "PHIN": "NPHI",
    # Resistivity (deep)
    "RT": "RT", "ILD": "RT", "AT90": "RT", "HDRS": "RT",
    # Sonic
    "DT": "DT", "DT24": "DT", "DTCO": "DT",
    # Photoelectric
    "PEF": "PEF", "PE": "PEF",
    # Caliper
    "CALI": "CALI", "CAL": "CALI", "HCAL": "CALI",
    # Spontaneous potential
    "SP": "SP",
}


def read_las(
    path: str | Path,
    canonicalise_curves: bool = True,
) -> WellLog:
    """
    Parse a LAS file and return a canonical WellLog.

    Parameters
    ----------
    path
        Path to a .las file (LAS 1.2, 2.0, 3.0 — lasio handles them).
    canonicalise_curves
        If True (default), rename curve columns to canonical names via
        CURVE_ALIASES. Set False to preserve raw mnemonics.

    Returns
    -------
    WellLog with .curves as a DataFrame indexed by depth (m).

    Raises
    ------
    FileNotFoundError, lasio.exceptions.LASHeaderError
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LAS file not found: {path}")

    las = lasio.read(str(path))

    # ── Header extraction ────────────────────────────────────────────────────
    well_block = {item.mnemonic: item.value for item in las.well}
    uwi    = well_block.get("UWI")  or well_block.get("API") or path.stem
    name   = well_block.get("WELL") or well_block.get("NAME")
    op     = well_block.get("COMP") or well_block.get("OPERATOR")

    # Location: LAS uses SLOC / X / Y / LATI / LONG inconsistently.
    lat = _to_float(well_block.get("LATI") or well_block.get("LAT"))
    lon = _to_float(well_block.get("LONG") or well_block.get("LON"))
    location = (lon, lat) if (lat is not None and lon is not None) else None

    kb_elev = _to_float(well_block.get("EKB") or well_block.get("KB"))

    # Depth range
    start = _to_float(well_block.get("STRT"))
    stop  = _to_float(well_block.get("STOP"))
    td_m  = stop if stop is not None else None

    # ── Curves → DataFrame ───────────────────────────────────────────────────
    curves_df = las.df()   # depth-indexed DataFrame
    if canonicalise_curves and not curves_df.empty:
        rename = {c: CURVE_ALIASES[c.upper()]
                  for c in curves_df.columns if c.upper() in CURVE_ALIASES}
        if rename:
            curves_df = curves_df.rename(columns=rename)

    # ── Formation tops (if present in OTHER section) ─────────────────────────
    tops = _parse_tops(las)

    return WellLog(
        uwi=str(uwi),
        name=name,
        operator=op,
        location=location,
        kb_elev_m=kb_elev,
        td_m=td_m,
        formation_tops=tops,
        curves=curves_df,
        source=f"LAS:{path.name}",
        raw={"start": start, "stop": stop, "version": las.version[0].value},
    )


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _to_float(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        f = float(v)
        # LAS often uses -999.25 as null sentinel
        if abs(f + 999.25) < 1e-3:
            return None
        return f
    except (TypeError, ValueError):
        return None


def _parse_tops(las) -> dict[str, float]:
    """
    Best-effort parse of formation tops from the OTHER section of a LAS file.
    LAS 3.0 has a dedicated TOPS section; LAS 2.0 puts them as free text.
    Returns {} if none found — that's fine, AGS/customer data fills this in.
    """
    tops: dict[str, float] = {}
    other = (las.other or "").strip()
    if not other:
        return tops
    for line in other.splitlines():
        # Heuristic: lines like "MANNVILLE   850.5   m" or "MANNVILLE: 850.5"
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(":", " ").replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            depth = float(parts[-1].rstrip("m"))
            name = " ".join(parts[:-1]).strip().upper()
            if name and 0 < depth < 10000:
                tops[name] = depth
        except ValueError:
            continue
    return tops


# ─── CLI / smoke ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m annix_intel.ingest.las <file.las>")
        sys.exit(2)
    log = read_las(sys.argv[1])
    import json
    print(json.dumps(log.summary(), indent=2, default=str))
    if log.curves is not None and not log.curves.empty:
        print("\nCurves preview:")
        print(log.curves.head())
