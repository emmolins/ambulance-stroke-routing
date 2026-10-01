"""
build_bay_area_grid.py
======================

Build the cell mask used by `scripts/grid_generator.jl`.

Reads the Bay Area ZIP-code shapefile (`data_prep/data/bayarea_zipcodes.shp`,
NAD83 / CA State Plane III feet), reprojects to WGS84, dissolves the 187 ZIP
polygons into a single Bay Area outline, and emits two artifacts that the
Julia grid generator consumes:

1.  ``sampled_points/bay_area_grid_cells.csv``
       One row per grid cell whose centre is inside the Bay Area polygon.
       Columns: cell_i, cell_j, lon_lo, lat_lo, lon_hi, lat_hi, lon_centre,
                lat_centre.

2.  ``sampled_points/bay_area_outline.geojson``
       The dissolved Bay Area polygon in WGS84, for use as an overlay in
       the heatmap figure.

Grid spec (matches the "Medium" choice — ~2 km cells over the Bay Area):
    GRID_SIZE = 100  (i ∈ 1..100, j ∈ 1..100)
    Bounding box = WGS84 bbox of the dissolved Bay Area polygon, padded
    slightly so cells don't get truncated at the edges.

Run::

    cd data_prep
    .venv/bin/python build_bay_area_grid.py

"""

import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

REPO_ROOT      = Path(__file__).resolve().parent.parent
SHAPEFILE_PATH = REPO_ROOT / "data_prep" / "data" / "bayarea_zipcodes.shp"
CELLS_CSV      = REPO_ROOT / "sampled_points" / "bay_area_grid_cells.csv"
OUTLINE_GEO    = REPO_ROOT / "sampled_points" / "bay_area_outline.geojson"

GRID_SIZE = 200   # matches original Hayward grid resolution; cells ~0.7 km × 0.9 km
PAD_DEG   = 0.01  # small padding so cells aren't truncated at the bbox edge


def main() -> None:
    print(f"Reading {SHAPEFILE_PATH} ...")
    gdf = gpd.read_file(SHAPEFILE_PATH)
    print(f"  {len(gdf)} ZIP polygons, CRS = {gdf.crs}")

    print("Reprojecting to WGS84 (EPSG:4326) ...")
    gdf = gdf.to_crs("EPSG:4326")

    print("Dissolving 187 ZIPs into one Bay Area polygon ...")
    bay_area = unary_union(gdf.geometry.values)
    minx, miny, maxx, maxy = bay_area.bounds
    print(f"  WGS84 bbox: lon [{minx:.4f}, {maxx:.4f}], "
          f"lat [{miny:.4f}, {maxy:.4f}]")
    print(f"  width  = {maxx-minx:.4f} deg")
    print(f"  height = {maxy-miny:.4f} deg")

    # Pad the bbox so the grid extends slightly past the polygon edges.
    lon_min, lon_max = minx - PAD_DEG, maxx + PAD_DEG
    lat_min, lat_max = miny - PAD_DEG, maxy + PAD_DEG
    lon_step = (lon_max - lon_min) / GRID_SIZE
    lat_step = (lat_max - lat_min) / GRID_SIZE

    print(f"\nBuilding {GRID_SIZE}×{GRID_SIZE} grid over padded bbox ...")
    print(f"  cell size ≈ {lon_step*111*0.79:.2f} km × {lat_step*111:.2f} km")
    print(f"  (longitude shrunk by cos(lat) ≈ 0.79 at lat ≈ 38°)")

    OUTLINE_GEO.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTLINE_GEO, "w") as f:
        json.dump({
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {"name": "Bay Area (dissolved ZIPs)"},
                "geometry": mapping(bay_area),
            }],
        }, f)
    print(f"  Wrote {OUTLINE_GEO}")

    # Build cell list — keep only cells whose CENTRE is inside the polygon.
    rows = []
    for i in range(1, GRID_SIZE + 1):
        lon_lo = lon_min + (i - 1) * lon_step
        lon_hi = lon_lo + lon_step
        lon_c  = (lon_lo + lon_hi) / 2
        for j in range(1, GRID_SIZE + 1):
            lat_lo = lat_min + (j - 1) * lat_step
            lat_hi = lat_lo + lat_step
            lat_c  = (lat_lo + lat_hi) / 2
            if bay_area.contains_properly(shape({
                "type": "Point", "coordinates": [lon_c, lat_c]
            })):
                rows.append({
                    "cell_i":     i,
                    "cell_j":     j,
                    "lon_lo":     lon_lo,
                    "lat_lo":     lat_lo,
                    "lon_hi":     lon_hi,
                    "lat_hi":     lat_hi,
                    "lon_centre": lon_c,
                    "lat_centre": lat_c,
                })

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(CELLS_CSV, index=False)
    pct = 100 * len(df) / (GRID_SIZE * GRID_SIZE)
    print(f"\nCells inside Bay Area : {len(df)} / {GRID_SIZE*GRID_SIZE}  ({pct:.1f}%)")
    print(f"  Wrote {CELLS_CSV}")
    print()
    print("Next:")
    print("  julia --project=. scripts/grid_generator.jl")


if __name__ == "__main__":
    main()
