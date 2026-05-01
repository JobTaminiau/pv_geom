"""Spatial join polygons↔tiles, group into tasks, handle tile-spanning polygons. M6 (PRD §6.2)."""

from __future__ import annotations


def assign_polygons_to_tiles(polygons, tile_index):
    """Return per-polygon (primary_tile_id, overlapping_tile_ids).

    Primary = tile containing the polygon centroid; overlap = any tile intersecting
    the polygon geometry. Used to ensure the per-tile-group worker fetches all tiles
    contributing returns and emits each row exactly once (from the primary).
    """
    raise NotImplementedError("M6")


def build_tile_groups(polygon_assignments):
    """Group polygons by primary tile and expand each task's tile-fetch list to the
    union of overlapping tiles for any polygon in the group."""
    raise NotImplementedError("M6")
