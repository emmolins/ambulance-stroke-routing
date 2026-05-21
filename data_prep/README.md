# data_prep — generating the patient sample points

This folder documents how the patient location datasets were generated for both regions. The canonical outputs used in the paper are **`../sampled_points/CA_points.csv`** (San Francisco Bay Area, 5,000 points) and **`../sampled_points/RI_points.csv`** (Rhode Island, 5,000 points). The notebooks reproduce the methodology; they do not overwrite the canonical files.

## What's in here

| File | What it is |
|---|---|
| `filter_points_CA.ipynb` | Bay Area sampling. Pop-weighted points → land filter against Berkeley ZIP shapefile. |
| `filter_points_RI.ipynb` | Rhode Island sampling. Pop-weighted points → land filter against TIGER/Line census tracts. |
| `requirements.txt` | Pinned Python deps shared by both notebooks. |
| `data/` | Where you place the downloaded raster + shapefile inputs (not committed; too large). |

## Why CA and RI use different shapefiles

The canonical `CA_points.csv` and `RI_points.csv` carry different per-region columns:

- **CA:** Natural Earth land polygon columns (`featurecla`, `scalerank`, `min_zoom`)
- **RI:** US Census Bureau TIGER/Line census-tract columns (`STATEFP`, `COUNTYFP`, `TRACTCE`, `GEOID`, ...)

The two notebooks preserve those choices to stay faithful to the original pipeline. The CA notebook additionally has a known divergence (it uses the Berkeley ZIP shapefile rather than Natural Earth — see the bottom of this README).

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

**Bay Area ZIP code shapefile (for the CA notebook):**

1. Visit https://geodata.lib.berkeley.edu/catalog/ark28722-s7888q
2. Download the full shapefile bundle (`.shp`, `.shx`, `.dbf`, `.prj`, etc. — keep them together).
3. Place them in `data_prep/data/` so the index lives at `data_prep/data/bayarea_zipcodes.shp`.

**Rhode Island census tract shapefile (for the RI notebook):**

1. Browse to https://www2.census.gov/geo/tiger/TIGER2020/TRACT/ (or whichever vintage year you prefer — TIGER2019, TIGER2021, etc. work the same way).
2. Download `tl_2020_44_tract.zip` (the `44` is Rhode Island's state FIPS code).
3. Unzip into `data_prep/data/` so the index lives at `data_prep/data/tl_2020_44_tract.shp`. If you use a different vintage, update the `RI_TRACTS_SHP` parameter in the RI notebook accordingly.

After setup, your `data/` should look like:

```
data_prep/data/
├── bayarea_zipcodes.shp
├── bayarea_zipcodes.shx
├── bayarea_zipcodes.dbf
├── bayarea_zipcodes.prj
├── tl_2020_44_tract.shp
├── tl_2020_44_tract.shx
├── tl_2020_44_tract.dbf
├── tl_2020_44_tract.prj
├── population_usa_2019-07-01.vrt
└── (population_*.tif tiles)
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

## Differences from the original Colab notebooks

Same updates applied to both notebooks:

1. **Paths** — `/content/...` (Colab-specific) replaced with relative paths under `data/`.
2. **Random seeds** — `np.random.seed(42)` and `random.seed(42)` set explicitly so the sampling is reproducible.
3. **Duplicate handling** — the original notebooks reported duplicate points but never dropped them (~1% of samples were duplicates). The updated notebooks drop them by default with `DROP_DUPLICATES = True`. Set this to `False` in the parameters cell to reproduce the original behavior exactly.
4. **API updates:**
   - `gpd.sjoin(..., op="intersects")` → `predicate="intersects"` (geopandas 0.14+).
   - `shape.unary_union` → `shape.union_all()` (shapely 2.x, used in the CA notebook's grid step).
5. **Known divergence (CA notebook):** the canonical `CA_points.csv` has columns `featurecla`, `scalerank`, `min_zoom` that come from a Natural Earth land polygon shapefile — not the Berkeley ZIP shapefile used here. The original final land-filter step appears to have used a different shapefile than the notebook documents. For now the CA notebook faithfully reproduces the **Colab logic** rather than reverse-engineering the final-step divergence. If you find or recreate the Natural Earth filter, update Step 4 accordingly and note it here. The RI notebook does not have this issue — its census-tract columns line up cleanly with the canonical `RI_points.csv` schema.

## Why this folder exists separately from the Julia code

The simulation code (`CA_simulations.jl` etc.) consumes CSVs from `sampled_points/`. The Python data-prep is a one-shot upstream step with completely different dependencies (geopandas, rasterio). Keeping it in its own folder with its own README and `requirements.txt` keeps the Julia environment lean and makes it obvious to reviewers that the two are separate concerns.
