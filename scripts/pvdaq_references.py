"""Start a reference table from the public PVDAQ inventory.

PVDAQ (NREL's PV Data Acquisition database, on the OEDI data lake) lists about
1,900 systems with a reported tilt and azimuth per mount. This script pulls the
systems inside a bounding box and writes one row per reported mount in the
format ``pv-geom compare-reference`` reads.

It does the clerical part only. Two columns are left blank on purpose, because
they are evidence a person has to establish:

``polygon_ids``
    Which polygons are this system. PVDAQ coordinates for PVOutput-derived
    systems are approximate (often tens of metres off, sometimes the wrong
    property), so the nearest polygon is not the answer. Match from imagery
    and module counts, and never by which polygons have agreeable angles.
``present_by`` / ``absent_at``
    Whether the array existed when the LiDAR was flown, from dated imagery.
    The reported start date is carried in ``notes`` as a lead, not as proof.

Usage:
    python scripts/pvdaq_references.py --bbox -113 32.5 -111 34.5 --out refs.csv
"""

from __future__ import annotations

import argparse
import io
import json
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://oedi-data-lake.s3.amazonaws.com/pvdaq/csv"
INVENTORY = f"{BASE}/systems_20250729.csv"
METADATA = BASE + "/system_metadata/{system_id}_system_metadata.json"


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read()


def _num(v: object) -> float | None:
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def mounts_of(system_id: int, row: pd.Series, fetch: bool) -> list[dict]:
    """The reported mounts of a system: from its metadata record when it can be
    fetched, else the single tilt/azimuth in the inventory."""
    meta: dict = {}
    if fetch:
        try:
            meta = json.loads(_get(METADATA.format(system_id=system_id)))
        except Exception as exc:
            print(f"  {system_id}: metadata not fetched ({exc}); using the inventory row")
    mounts = [{"name": k, "tilt": _num(m.get("tilt")), "azimuth": _num(m.get("azimuth")),
               "tracking": str(m.get("tracking", "")).lower() in ("t", "true")}
              for k, m in (meta.get("Mount") or {}).items()]
    if not mounts:
        mounts = [{"name": "Mount 0", "tilt": _num(row.get("tilt")),
                   "azimuth": _num(row.get("azimuth")),
                   "tracking": str(row.get("tracking", "")).lower() == "tracking"}]
    started = (meta.get("System") or {}).get("started_on") or row.get("first_timestamp")
    modules = sum(int(m["quantity"]) for m in (meta.get("Modules") or {}).values()
                  if str(m.get("quantity", "")).isdigit())
    for m in mounts:
        m["started"], m["modules"] = started, modules or None
    return mounts


def build(inventory: pd.DataFrame, bbox: tuple[float, float, float, float],
          fetch: bool = True) -> pd.DataFrame:
    lon0, lat0, lon1, lat1 = bbox
    lat = pd.to_numeric(inventory["latitude"], errors="coerce")
    lon = pd.to_numeric(inventory["longitude"], errors="coerce")
    inside = inventory[lat.between(lat0, lat1) & lon.between(lon0, lon1)]
    rows = []
    for _, sysrow in inside.iterrows():
        sid = int(sysrow["system_id"])
        pvoutput = "pvoutput" in str(sysrow["system_public_name"]).lower()
        mounts = mounts_of(sid, sysrow, fetch)
        for i, m in enumerate(mounts):
            if m["tracking"] or m["tilt"] is None:
                continue                      # no fixed geometry to compare
            azimuth = m["azimuth"] if m["azimuth"] is not None and m["azimuth"] >= 0 else None
            # PVOutput takes orientation as a compass point and tilt in whole
            # degrees; 1 degree on such a record is nearly always a default.
            placeholder = pvoutput and m["tilt"] <= 1.0
            rows.append({
                "reference_id": f"pvdaq-{sid}-m{i}", "polygon_ids": "",
                "tilt_deg": m["tilt"], "azimuth_deg": azimuth, "system_id": sid,
                "source": f"PVDAQ {sid}: {sysrow['system_public_name']}",
                "scope": "mount" if len(mounts) > 1 else "system",
                "grade": "self_reported" if pvoutput else "documented",
                "tilt_resolution_deg": 1.0,
                "azimuth_resolution_deg": 45.0 if pvoutput else 1.0,
                "suspect": placeholder, "present_by": "", "absent_at": "",
                "notes": f"reported at {sysrow['latitude']}, {sysrow['longitude']} "
                         f"({sysrow['site_location']}); reported start {m['started']}; "
                         f"modules {m['modules'] or 'n/a'}; mount type {sysrow.get('type')}",
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bbox", nargs=4, type=float, required=True,
                    metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--inventory", help="Local copy of the PVDAQ systems CSV (else downloaded).")
    ap.add_argument("--no-metadata", action="store_true",
                    help="Do not fetch per-system records; one mount per system.")
    args = ap.parse_args()
    raw = Path(args.inventory).read_bytes() if args.inventory else _get(INVENTORY)
    refs = build(pd.read_csv(io.BytesIO(raw), low_memory=False), tuple(args.bbox),
                 fetch=not args.no_metadata)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    refs.to_csv(args.out, index=False)
    n_ok = int((~refs["suspect"]).sum()) if len(refs) else 0
    print(f"{len(refs)} reference mounts written to {args.out} "
          f"({n_ok} not flagged as placeholders). Fill polygon_ids and present_by/absent_at.")


if __name__ == "__main__":
    main()
