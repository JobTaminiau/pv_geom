"""Spatial join polygons↔tiles + tile-group partitioner. M6 (PRD §6.2).

Each polygon is assigned to a *primary* tile (the tile containing its centroid)
and to a *set* of *overlapping* tiles (any tile its geometry intersects). Tile
groups are formed by primary-tile assignment; each group's tile-fetch list is
the union of all overlapping tiles for any polygon in the group, so per-tile
point sets can be merged in-process during fitting without a Dask shuffle.
The primary-tile worker is the only one that emits an output row, eliminating
duplicates without a separate dedup pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import pandas as pd


@dataclass(frozen=True)
class TileGroup:
    """One unit of work for the per-tile-group Dask task."""

    primary_tile_id: str
    polygon_ids: tuple[str, ...]
    fetch_tile_ids: tuple[str, ...]      # union of overlapping tiles for the group


def assign_polygons_to_tiles(
    polygons: gpd.GeoDataFrame,
    tile_index: gpd.GeoDataFrame,
    *,
    polygon_id_col: str = "polygon_id",
    tile_id_col: str = "Name",
) -> pd.DataFrame:
    """Compute (polygon_id, primary_tile_id, overlapping_tile_ids).

    ``polygons`` and ``tile_index`` must be in the **same** CRS. The primary
    tile is the one whose geometry contains the polygon centroid (deterministic
    even when centroids fall on tile boundaries — the first matching tile by
    spatial-index iteration order wins; in practice tile boundaries are sets
    of measure zero). Overlapping tiles include the primary plus any tile
    whose geometry intersects the polygon.
    """
    if polygons.crs is None or tile_index.crs is None or polygons.crs != tile_index.crs:
        raise ValueError(
            "polygons and tile_index must share a CRS; "
            f"got {polygons.crs} and {tile_index.crs}"
        )

    centroids = polygons.geometry.centroid

    # Primary tiles: tile.contains(polygon.centroid).
    primary = gpd.sjoin(
        gpd.GeoDataFrame(
            {polygon_id_col: polygons[polygon_id_col]},
            geometry=centroids,
            crs=polygons.crs,
        ),
        tile_index[[tile_id_col, "geometry"]].rename(columns={tile_id_col: "_tid"}),
        predicate="within",
        how="left",
    )
    # Drop duplicates where centroid is on a boundary (within multiple tiles).
    primary = (
        primary.dropna(subset=["_tid"])
        .drop_duplicates(subset=polygon_id_col, keep="first")
        [[polygon_id_col, "_tid"]]
        .rename(columns={"_tid": "primary_tile_id"})
    )

    # Overlapping tiles: any tile that intersects the polygon.
    overlap = gpd.sjoin(
        polygons[[polygon_id_col, "geometry"]],
        tile_index[[tile_id_col, "geometry"]].rename(columns={tile_id_col: "_tid"}),
        predicate="intersects",
        how="left",
    )
    overlap = overlap.dropna(subset=["_tid"])
    overlap_lists = (
        overlap.groupby(polygon_id_col)["_tid"]
        .apply(lambda s: tuple(sorted(set(s))))
        .reset_index()
        .rename(columns={"_tid": "overlapping_tile_ids"})
    )

    out = primary.merge(overlap_lists, on=polygon_id_col, how="left")
    out["overlapping_tile_ids"] = out["overlapping_tile_ids"].apply(
        lambda x: tuple(x) if isinstance(x, tuple) else (x,) if isinstance(x, str) else ()
    )
    return out.reset_index(drop=True)


def build_tile_groups(assignments: pd.DataFrame) -> list[TileGroup]:
    """Group polygons by primary tile; expand each task's fetch list to the
    union of overlapping tiles for any polygon in the group."""
    groups: list[TileGroup] = []
    for primary, sub in assignments.groupby("primary_tile_id", sort=True):
        fetch_ids: set[str] = set()
        for tids in sub["overlapping_tile_ids"]:
            fetch_ids.update(tids)
        fetch_ids.add(str(primary))
        polygon_ids = tuple(sub["polygon_id"].astype(str).tolist())
        groups.append(
            TileGroup(
                primary_tile_id=str(primary),
                polygon_ids=polygon_ids,
                fetch_tile_ids=tuple(sorted(fetch_ids)),
            )
        )
    return groups
