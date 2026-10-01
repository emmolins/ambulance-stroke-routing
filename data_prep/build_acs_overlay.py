"""
build_acs_overlay.py
====================

Download the demographic + urbanicity overlay needed for the equity analysis.

Pulls three sources, joins them, emits two artifacts:

Sources
-------
1. TIGER 2020 California tract polygons
       https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_06_tract.zip
2. ACS 2022 5-year tract-level demographics (Census Bureau API, no key needed
   for a single small request)
       https://api.census.gov/data/2022/acs/acs5
3. USDA ERS RUCA 2010 (revised) tract-level urbanicity codes
       https://www.ers.usda.gov/sites/default/files/_laserfiche/DataFiles/53241/ruca2010revised.xlsx

All three are filtered to the nine Bay Area counties:
    Alameda (001), Contra Costa (013), Marin (041), Napa (055),
    San Francisco (075), San Mateo (081), Santa Clara (085),
    Solano (095), Sonoma (097)

Outputs
-------
1. data_prep/data/bay_area_tracts.geojson
       Tract polygons in WGS84, used by scripts/grid_to_tract_join.py to map
       each grid cell to a tract.

2. sampled_points/bay_area_tract_demographics.csv
       Flat table indexed by GEOID, with the following columns:
           GEOID            — 11-digit Census tract identifier
           county_fips      — 5-digit county FIPS
           tract_name       — human-readable name
           total_pop        — total population (ACS B01003_001E)
           median_hh_income — median household income, $ (ACS B19013_001E)
           pct_below_pov    — % of pop in poverty (ACS B17001_002E / _001E)
           ruca_code        — USDA RUCA 2010 code (1–10)
           urbanicity       — 'urban' (RUCA 1–3) or 'rural' (RUCA 4–10)

Run
---
    cd data_prep
    .venv/bin/pip install -r requirements.txt   # if not already
    .venv/bin/python build_acs_overlay.py

The script caches downloads in `data_prep/data/cache/`, so reruns are fast.
"""

import io
import json
import os
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = REPO_ROOT / "data_prep" / "data"
CACHE_DIR  = DATA_DIR / "cache"
OUT_TRACTS = DATA_DIR / "bay_area_tracts.geojson"
OUT_TABLE  = REPO_ROOT / "sampled_points" / "bay_area_tract_demographics.csv"

# 9-county Bay Area FIPS codes (state 06 = California).
BAY_COUNTY_FIPS = ["001", "013", "041", "055", "075", "081", "085", "095", "097"]
COUNTY_NAMES = {
    "001": "Alameda", "013": "Contra Costa", "041": "Marin", "055": "Napa",
    "075": "San Francisco", "081": "San Mateo", "085": "Santa Clara",
    "095": "Solano",  "097": "Sonoma",
}

TIGER_URL = ("https://www2.census.gov/geo/tiger/TIGER2020/TRACT/"
             "tl_2020_06_tract.zip")
ACS_VARS  = {
    "B01003_001E": "total_pop",
    "B19013_001E": "median_hh_income",
    "B17001_002E": "pop_below_pov",
    "B17001_001E": "pop_pov_universe",
}
ACS_URL = (
    "https://api.census.gov/data/2022/acs/acs5"
    f"?get=NAME,{','.join(ACS_VARS)}"
    "&for=tract:*"
    f"&in=state:06%20county:{','.join(BAY_COUNTY_FIPS)}"
)
RUCA_URL = ("https://www.ers.usda.gov/sites/default/files/_laserfiche/"
            "DataFiles/53241/ruca2010revised.xlsx")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fetch(url: str, dest: Path) -> Path:
    """Cached HTTP download — returns local path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached  {dest.name}  ({dest.stat().st_size / 1024:.0f} KB)")
        return dest
    print(f"  fetching  {url}")
    # Census/ERS sometimes block default urllib UA; spoof a browser one.
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())
    print(f"  → {dest}  ({dest.stat().st_size / 1024:.0f} KB)")
    return dest


# ---------------------------------------------------------------------------
# Step 1 — TIGER tract polygons
# ---------------------------------------------------------------------------
def load_bay_area_tracts() -> gpd.GeoDataFrame:
    print("[1/3] TIGER 2020 California tract polygons")
    zpath = fetch(TIGER_URL, CACHE_DIR / "tl_2020_06_tract.zip")
    with zipfile.ZipFile(zpath) as zf:
        # Extract once to a stable directory so geopandas can sidecar-read.
        extract_dir = CACHE_DIR / "tl_2020_06_tract"
        if not (extract_dir / "tl_2020_06_tract.shp").exists():
            extract_dir.mkdir(exist_ok=True)
            zf.extractall(extract_dir)

    gdf = gpd.read_file(extract_dir / "tl_2020_06_tract.shp")
    print(f"  CA tracts loaded     : {len(gdf)}")
    gdf = gdf[gdf["COUNTYFP"].isin(BAY_COUNTY_FIPS)].copy()
    gdf = gdf.to_crs("EPSG:4326")
    gdf["county_name"] = gdf["COUNTYFP"].map(COUNTY_NAMES)
    print(f"  Bay Area tracts kept : {len(gdf)}")
    return gdf[["GEOID", "COUNTYFP", "county_name", "NAMELSAD", "geometry"]]


# ---------------------------------------------------------------------------
# Step 2 — ACS demographics
# ---------------------------------------------------------------------------
def _require_census_key() -> str:
    """
    The Census Bureau requires an API key for all data requests. Free,
    instant signup at https://api.census.gov/data/key_signup.html — you
    get the key by email within ~1 minute.

    Set it before running:
        export CENSUS_API_KEY=your-key-here
        .venv/bin/python build_acs_overlay.py
    """
    key = os.environ.get("CENSUS_API_KEY")
    if not key:
        raise RuntimeError(
            "CENSUS_API_KEY environment variable is not set.\n\n"
            "The Census API requires a free key for all requests.\n"
            "1. Sign up: https://api.census.gov/data/key_signup.html\n"
            "   (Takes ~1 minute — they email you the key.)\n"
            "2. Activate by clicking the link in the email.\n"
            "3. Set the key in your shell:\n"
            "       export CENSUS_API_KEY=your-key-here\n"
            "4. Re-run this script.\n"
        )
    return key


def _fetch_acs_county(state_fp: str, county_fp: str, api_key: str) -> list:
    """
    Fetch ACS rows for one county. The Census API is finicky about multi-
    hierarchy `in=` clauses, so we query county-by-county and concatenate.
    """
    var_str = ",".join(ACS_VARS)
    url = ("https://api.census.gov/data/2022/acs/acs5"
           f"?get=NAME,{var_str}"
           "&for=tract:*"
           f"&in=state:{state_fp}&in=county:{county_fp}"
           f"&key={api_key}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read().decode()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        # Show what the API actually returned so the user can diagnose.
        # Strip the key from the URL so it's not echoed in error output.
        safe_url = url.replace(api_key, "<REDACTED>")
        raise RuntimeError(
            f"Census API did not return JSON for county {county_fp}.\n"
            f"URL: {safe_url}\n"
            f"Status: {r.status}\n"
            f"Body (first 500 chars):\n{body[:500]}"
        )


def load_acs() -> pd.DataFrame:
    print("[2/3] ACS 2022 5-year tract demographics")
    cache = CACHE_DIR / "acs2022_bay_area_tracts.json"

    if cache.exists() and cache.stat().st_size > 0:
        print(f"  cached  {cache.name}")
        all_rows = json.loads(cache.read_text())
    else:
        # Query each county separately; the multi-county hierarchy clause is
        # unreliable in the Census API.
        api_key = _require_census_key()
        all_rows = None
        for cfp in BAY_COUNTY_FIPS:
            print(f"  fetching county {cfp} ({COUNTY_NAMES[cfp]}) ...")
            chunk = _fetch_acs_county("06", cfp, api_key)
            header, *rows = chunk
            if all_rows is None:
                all_rows = [header] + rows
            else:
                all_rows.extend(rows)
        cache.write_text(json.dumps(all_rows))

    header, *rows = all_rows
    df = pd.DataFrame(rows, columns=header)
    df = df.rename(columns={**ACS_VARS, "NAME": "tract_name"})

    # ACS returns suppressed values as negative codes (e.g. -666666666); coerce.
    for col in ACS_VARS.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < 0, col] = pd.NA

    df["GEOID"] = df["state"] + df["county"] + df["tract"]
    df["pct_below_pov"] = (df["pop_below_pov"] / df["pop_pov_universe"]) * 100

    print(f"  rows received        : {len(df)}")
    if df["median_hh_income"].notna().any():
        print(f"  median income range  : "
              f"${df['median_hh_income'].min():,.0f} – "
              f"${df['median_hh_income'].max():,.0f}")
    return df[["GEOID", "tract_name", "total_pop", "median_hh_income",
               "pct_below_pov"]]


# ---------------------------------------------------------------------------
# Step 3 — USDA RUCA
# ---------------------------------------------------------------------------
def load_ruca() -> pd.DataFrame:
    print("[3/3] USDA ERS RUCA 2010 (revised) tract codes")
    xpath = fetch(RUCA_URL, CACHE_DIR / "ruca2010revised.xlsx")

    # The RUCA file has an errata note + blank rows above the real header.
    # Identify the header row by: many non-NaN cells AND the row contains
    # both "RUCA" and "FIPS" in its values. The errata row has "RUCA" but
    # not "FIPS", and only 1 non-NaN cell.
    raw = pd.read_excel(xpath, sheet_name=0, header=None, dtype=str)
    header_row = None
    for i in range(min(15, len(raw))):
        row = raw.iloc[i].astype(str)
        cells = [c.strip() for c in row if isinstance(c, str) and c.strip()
                 and c.lower() != "nan"]
        if len(cells) < 5:
            continue
        joined = " | ".join(cells).lower()
        if "ruca" in joined and "fips" in joined:
            header_row = i
            break
    if header_row is None:
        raise RuntimeError("Could not locate RUCA header row.\n"
                           f"First 10 rows:\n{raw.head(10).to_string()}")

    ruca = pd.read_excel(xpath, sheet_name=0, header=header_row, dtype=str)
    ruca.columns = [c.strip() for c in ruca.columns]

    # Find the 11-digit tract FIPS column. The file ALSO has a 5-digit
    # county FIPS column, which must NOT be picked.
    geoid_col = next((c for c in ruca.columns
                      if "tract" in c.lower() and "fips" in c.lower()), None)
    if geoid_col is None:
        # Older RUCA vintages sometimes label it "Census Tract" or "Geocode".
        geoid_col = next((c for c in ruca.columns
                          if re.search(r"State.*County.*Tract|tract.*code|geoid",
                                        c, re.I)), None)

    ruca_col = next((c for c in ruca.columns
                     if re.search(r"^Primary RUCA|^RUCA(?!.*Secondary)|^RUCA1",
                                   c, re.I)), None)
    if geoid_col is None or ruca_col is None:
        raise RuntimeError(
            f"Could not detect RUCA columns. Available: {list(ruca.columns)}")
    print(f"  GEOID column         : '{geoid_col}'")
    print(f"  RUCA column          : '{ruca_col}'")

    ruca = ruca.rename(columns={geoid_col: "GEOID", ruca_col: "ruca_code"})
    ruca["GEOID"] = ruca["GEOID"].astype(str).str.zfill(11)
    ruca = ruca[ruca["GEOID"].str.startswith("06")]    # CA only
    ruca["ruca_code"] = pd.to_numeric(ruca["ruca_code"], errors="coerce")

    ruca["urbanicity"] = ruca["ruca_code"].apply(
        lambda c: "urban" if c is not None and c <= 3 else (
                  "rural" if c is not None and c >= 4 else pd.NA))

    print(f"  CA RUCA rows         : {len(ruca)}")
    return ruca[["GEOID", "ruca_code", "urbanicity"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("="*70)
    print("BAY AREA EQUITY OVERLAY BUILD")
    print("="*70)

    tracts = load_bay_area_tracts()
    acs    = load_acs()
    ruca   = load_ruca()
    print()

    # ----- Merge --------------------------------------------------------
    print("Merging ACS + RUCA into tract table ...")
    merged = (tracts
              .merge(acs,  on="GEOID", how="left")
              .merge(ruca, on="GEOID", how="left"))

    flat = merged.drop(columns="geometry").rename(
        columns={"COUNTYFP": "county_fips", "NAMELSAD": "tract_label"})

    # Diagnostic: how much data is complete?
    n = len(flat)
    n_inc  = flat["median_hh_income"].notna().sum()
    n_ruca = flat["ruca_code"].notna().sum()
    print(f"  Tracts in table      : {n}")
    print(f"  With income          : {n_inc} ({100*n_inc/n:.1f}%)")
    print(f"  With RUCA code       : {n_ruca} ({100*n_ruca/n:.1f}%)")

    # ----- Write --------------------------------------------------------
    OUT_TRACTS.parent.mkdir(parents=True, exist_ok=True)
    merged[["GEOID", "county_name", "geometry"]].to_file(OUT_TRACTS,
                                                          driver="GeoJSON")
    print(f"  ✓ Wrote {OUT_TRACTS}")

    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    flat.to_csv(OUT_TABLE, index=False)
    print(f"  ✓ Wrote {OUT_TABLE}")

    print()
    print("Next, AFTER the grid generator finishes:")
    print("  python scripts/grid_to_tract_join.py")
    print("  julia --project=. scripts/equity_analysis.jl")


if __name__ == "__main__":
    sys.exit(main())
