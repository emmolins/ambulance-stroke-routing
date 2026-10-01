# data_prep — generating the patient sample points

This folder documents how the patient location datasets were generated for both regions. The canonical outputs used in the paper are **`../sampled_points/CA_points.csv`** (San Francisco Bay Area, 5,000 points) and **`../sampled_points/RI_points.csv`** (Rhode Island, 5,000 points). The notebooks reproduce the methodology; they do not overwrite the canonical files.

## What's in here

| File | What it is |
|---|---|
| `filter_points_CA.ipynb` | Bay Area sampling. Pop-weighted points → land filter against Natural Earth land polygons (Berkeley ZIP shapefile is used to bound the raster mask). |
| `filter_points_RI.ipynb` | Rhode Island sampling. Pop-weighted points → land filter against TIGER/Line census tracts. |
| `requirements.txt` | Pinned Python deps shared by both notebooks. |
| `data/` | Where you place the downloaded raster + shapefile inputs (not committed; too large). |

## Why CA and RI use different shapefiles

The canonical `CA_points.csv` and `RI_points.csv` carry different per-region columns:

- **CA:** Natural Earth land polygon columns (`featurecla`, `scalerank`, `min_zoom`) — the final land filter uses a Natural Earth global land shapefile. The Berkeley ZIP shapefile is used earlier in the pipeline (Step 1) to mask the population raster to the Bay Area before sampling.
- **RI:** US Census Bureau TIGER/Line census-tract columns (`STATEFP`, `COUNTYFP`, `TRACTCE`, `GEOID`, ...) — the census-tract polygons act as both the region bound and the final land filter.

Both notebooks now reproduce the methodology that produced the canonical CSV column schemas.

## One-time setup

### 1. Create a Python environment

We recommend a project-local venv so the geospatial deps don't conflict with anything else:

```sh
cd data_prep
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=stroke-routing-data-prep
```

### 2. Download the input data

Both files go in `data_prep/data/` (the folder is created automatically when you make the venv, or you can `mkdir -p data` yourself):

**Population density raster (~several GB after extracting the tiles):**

1. Visit https://data.humdata.org/dataset/united-states-high-resolution-population-density-maps-demographic-estimates
2. Download the **`general` (total population) 30-m tiles** for the United States, July 2019 vintage. You'll get a set of GeoTIFFs and a `.vrt` index.
3. Place all the files in `data_prep/data/` so the index file lives at `data_prep/data/population_usa_2019-07-01.vrt`.

> Note: the original paper used Meta/CIESIN's 2019 release. Meta/CIESIN have since paused new releases — the 2019 dataset remains the most recent one for U.S. high-resolution population. If a newer release is available by the time you re-run, document the change in your manuscript's methods.

**Bay Area ZIP code shapefile (for the CA notebook — region mask in Step 1):**

1. Visit https://geodata.lib.berkeley.edu/catalog/ark28722-s7888q
2. Download the full shapefile bundle (`.shp`, `.shx`, `.dbf`, `.prj`, etc. — keep them together).
3. Place them in `data_prep/data/` so the index lives at `data_prep/data/bayarea_zipcodes.shp`.

**Natural Earth 10m land polygons (for the CA notebook — final land filter in Step 4):**

1. Visit https://www.naturalearthdata.com/downloads/10m-physical-vectors/
2. Find **"Land"** and click **"Download land"** to get `ne_10m_land.zip`.
3. Unzip into a subfolder so the file lives at `data_prep/data/ne_10m_land/ne_10m_land.shp` (keep all the `ne_10m_land.*` companion files together in that folder).

**Rhode Island census tract shapefile (for the RI notebook):**

1. Browse to https://www2.census.gov/geo/tiger/TIGER2020/TRACT/ (or whichever vintage year you prefer — TIGER2019, TIGER2021, etc. work the same way).
2. Download `tl_2020_44_tract.zip` (the `44` is Rhode Island's state FIPS code).
3. Unzip into `data_prep/data/` so the index lives at `data_prep/data/tl_2020_44_tract.shp`. If you use a different vintage, update the `RI_TRACTS_SHP` parameter in the RI notebook accordingly.

After setup, your `data/` should look like:

```
data_prep/data/
├── bayarea_zipcodes.shp (+ .shx, .dbf, .prj, ...)
├── tl_2020_44_tract.shp (+ .shx, .dbf, .prj, ...)
├── ne_10m_land/
│   └── ne_10m_land.shp (+ .shx, .dbf, .prj, ...)
├── population_usa_2019-07-01.vrt
└── (population_usa*_*.tif tiles — at minimum 28_-130, 38_-130, 38_-80 for the regions used)
```

### 3. Launch Jupyter

```sh
cd data_prep
source .venv/bin/activate
jupyter notebook
```

Open either notebook and run all cells. Pick the **"stroke-routing-data-prep"** kernel.

## What the notebooks produce

All outputs are written to `../sampled_points/` and are **separately named** from the canonical paper data so the originals are never overwritten:

| File | Source notebook | Description |
|---|---|---|
| `sampled_points/CA_points_raw.csv` | `filter_points_CA.ipynb` | Bay Area sample before land filter. |
| `sampled_points/CA_points_regenerated.csv` | `filter_points_CA.ipynb` | Bay Area sample after land filter. |
| `sampled_points/RI_points_raw.csv` | `filter_points_RI.ipynb` | Rhode Island sample before land filter. |
| `sampled_points/RI_points_regenerated.csv` | `filter_points_RI.ipynb` | Rhode Island sample after land filter. |

The canonical `CA_points.csv` and `RI_points.csv` are left untouched. To compare:

```python
import pandas as pd
canonical = pd.read_csv("../sampled_points/RI_points.csv")
regenerated = pd.read_csv("../sampled_points/RI_points_regenerated.csv")
print(f"Canonical: {len(canonical)} points; Regenerated: {len(regenerated)} points")
```

## Reproducibility notes

- **Random seeds.** Both notebooks call `np.random.seed(42)` and `random.seed(42)` in the parameters cell so re-runs produce identical samples.
- **Duplicate handling.** Because the population raster is gridded (~30 m cells) and `np.random.choice` samples with replacement, ~1% of draws collide on the same grid cell and produce identical lat/lon coordinates. The notebooks **keep these duplicates by default** (`DROP_DUPLICATES = False`) — they are a legitimate outcome of population-weighted sampling with replacement, and the canonical CSVs were generated this way. Set the flag to `True` only if your downstream use specifically requires unique coordinates.

## Why this folder exists separately from the Julia code

The simulation code (`CA_simulations.jl` etc.) consumes CSVs from `sampled_points/`. The Python data-prep is a one-shot upstream step with completely different dependencies (geopandas, rasterio). Keeping it in its own folder with its own README and `requirements.txt` keeps the Julia environment lean and makes it obvious to reviewers that the two are separate concerns.
