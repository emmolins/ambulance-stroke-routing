"""
scripts/grid_to_tract_join.py
=============================

Spatially join per-patient grid outcomes to Census tracts, then merge with
ACS + RUCA demographics. Produces the equity-analysis-ready table.

Inputs
------
1. sampled_points/CA_grid_patients.csv
       From scripts/grid_aggregate.jl. One row per simulated patient with
       (cell_i, cell_j, start_lat, start_lon, stroke_type, all four policies'
       action/reward/travel_time).

2. data_prep/data/bay_area_tracts.geojson
       From data_prep/build_acs_overlay.py. Tract polygons in WGS84.

3. sampled_points/bay_area_tract_demographics.csv
       From data_prep/build_acs_overlay.py. Tract demographics + RUCA.

Output
------
sampled_points/CA_grid_patients_with_demographics.csv
    One row per patient with all of the above plus:
        GEOID          — Census tract identifier
        county_name    — Bay Area county
        median_hh_income, pct_below_pov, total_pop
        ruca_code, urbanicity

Run
---
    .venv/bin/python scripts/grid_to_tract_join.py
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

REPO_ROOT = Path(__file__).resolve().parent.parent
PATIENTS_CSV = REPO_ROOT / "sampled_points" / "CA_grid_patients.csv"
TRACTS_GEO   = REPO_ROOT / "data_prep" / "data" / "bay_area_tracts.geojson"
TRACT_TABLE  = REPO_ROOT / "sampled_points" / "bay_area_tract_demographics.csv"
OUT_CSV      = REPO_ROOT / "sampled_points" / "CA_grid_patients_with_demographics.csv"


def main() -> None:
    for f in (PATIENTS_CSV, TRACTS_GEO, TRACT_TABLE):
        if not f.exists():
            raise FileNotFoundError(
                f"Missing prerequisite: {f}\n"
                "Order of operations:\n"
                "  1. data_prep/build_bay_area_grid.py  (cell mask)\n"
                "  2. data_prep/build_acs_overlay.py    (demographics)\n"
                "  3. scripts/grid_generator.jl         (simulations — ~35-50h)\n"
                "  4. scripts/grid_aggregate.jl         (pool cells)\n"
                "  5. THIS SCRIPT")

    print("Reading inputs ...")
    patients = pd.read_csv(PATIENTS_CSV)
    tracts   = gpd.read_file(TRACTS_GEO)
    demo     = pd.read_csv(TRACT_TABLE, dtype={"GEOID": str, "county_fips": str})
    print(f"  patients : {len(patients):,}")
    print(f"  tracts   : {len(tracts)}")
    print(f"  demo rows: {len(demo)}")
    print()

    # Convert patients to a GeoDataFrame for the spatial join.
    print("Building patient-point GeoDataFrame (WGS84) ...")
    geom = [Point(lon, lat) for lon, lat in zip(patients.start_lon,
                                                  patients.start_lat)]
    p_gdf = gpd.GeoDataFrame(patients, geometry=geom, crs="EPSG:4326")

    # Project both to a metric CRS so the within-test is fast and exact.
    # California Albers (EPSG:3310) is the standard CA-wide projection.
    print("Reprojecting to CA Albers (EPSG:3310) for spatial join ...")
    p_gdf  = p_gdf.to_crs("EPSG:3310")
    tracts = tracts.to_crs("EPSG:3310")

    print("Spatial join (point-in-polygon) ...")
    joined = gpd.sjoin(p_gdf, tracts[["GEOID", "geometry"]],
                       how="left", predicate="within")
    joined = joined.drop(columns=["geometry", "index_right"])

    n_unmatched = joined["GEOID"].isna().sum()
    print(f"  patients matched : {(len(joined) - n_unmatched):,}")
    print(f"  patients on edge : {n_unmatched:,} (point fell outside all tracts)")

    # Drop edge cases — they're inside the grid bbox but outside any tract.
    joined = joined.dropna(subset=["GEOID"]).copy()
    joined["GEOID"] = joined["GEOID"].astype(str)

    print("\nMerging with tract demographics ...")
    final = joined.merge(demo, on="GEOID", how="left")
    print(f"  final rows       : {len(final):,}")
    print(f"  with income      : {final['median_hh_income'].notna().sum():,}")
    print(f"  with RUCA        : {final['ruca_code'].notna().sum():,}")

    final.to_csv(OUT_CSV, index=False)
    print(f"\n  ✓ Wrote {OUT_CSV}")
    print()
    print("Next:")
    print("  julia --project=. scripts/equity_analysis.jl")


if __name__ == "__main__":
    main()
