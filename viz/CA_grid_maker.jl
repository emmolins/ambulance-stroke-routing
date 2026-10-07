# =============================================================================
# viz/CA_grid_maker.jl
#
# Geospatial heatmaps of stroke-patient outcome under each routing policy, and
# pairwise policy-difference maps, over the nine-county Bay Area.
#
# Consumes the pooled per-cell summary produced by scripts/grid_aggregate.jl:
#   sampled_points/CA_grid_cell_means.csv
#       columns: cell_i, cell_j, cell_lat, cell_lon, n_samples,
#                reward_optimal, reward_nearest, reward_heur1, reward_heur2,
#                diff_opt_minus_nearest, diff_opt_minus_heur1,
#                diff_opt_minus_heur2, diff_heur1_minus_nearest,
#                diff_heur2_minus_nearest
#
# The grid extent comes from sampled_points/bay_area_grid_cells.csv (produced
# by data_prep/build_bay_area_grid.py), so the same plotting code adapts to
# whatever grid resolution / region scope was set there.
#
# Outputs (PDF) — written to simulation_results/:
#   CA_outcome_optimal.pdf          MDP optimal policy outcome
#   CA_outcome_nearest.pdf          Nearest-hospital baseline
#   CA_outcome_heur1.pdf            Heuristic 1
#   CA_outcome_heur2.pdf            Heuristic 2
#   CA_diff_opt_minus_nearest.pdf   MDP advantage over nearest (diverging cmap)
#   CA_diff_opt_minus_heur1.pdf     MDP advantage over Heuristic 1
#   CA_diff_heur1_minus_nearest.pdf Heuristic 1 advantage over nearest
#
# Difference maps use a diverging (blue→white→red) colormap symmetric around
# zero — NEGATIVE cells (where the comparator wins) are visible, not clipped.
#
# Run
#   julia --project=. viz/CA_grid_maker.jl
# =============================================================================

using CSV
using DataFrames
using GMT
using Statistics

const CELL_MEAN_CSV  = "sampled_points/CA_grid_cell_means.csv"
const GRID_CELLS_CSV = "sampled_points/bay_area_grid_cells.csv"
const OUT_DIR        = "simulation_results"

# Bay Area cities annotated on every map. Filtered to those inside the
# plotted region at render time.
const CITIES = [
    (-122.4194, 37.7749, "San Francisco"),
    (-121.8863, 37.3382, "San Jose"),
    (-122.2711, 37.8044, "Oakland"),
    (-122.2585, 37.8716, "Berkeley"),
    (-121.9886, 37.5483, "Fremont"),
    (-122.0808, 37.6688, "Hayward"),
    (-121.8058, 38.0049, "Antioch"),
    (-121.7680, 37.6819, "Livermore"),
    (-122.1430, 37.4419, "Palo Alto"),
    (-122.7140, 38.4404, "Santa Rosa"),
    (-122.5311, 37.9735, "San Rafael"),
    (-122.2867, 37.8716, "Vallejo"),
    (-122.0322, 37.9577, "Concord"),
    (-121.2908, 38.2497, "Fairfield"),
    (-122.4596, 37.5630, "San Mateo"),
    (-121.5680, 37.0058, "Gilroy"),
]

# ============================================================================
# Load cell-mean table + derive region bounds from the grid mask
# ============================================================================
isfile(CELL_MEAN_CSV) ||
    error("$CELL_MEAN_CSV not found. Run scripts/grid_aggregate.jl first " *
          "(which itself needs scripts/grid_generator.jl to have finished).")

println("Reading $CELL_MEAN_CSV ...")
cell = CSV.read(CELL_MEAN_CSV, DataFrame)
println("  $(nrow(cell)) cells, $(ncol(cell)) columns")

# Region bounds come from the grid mask (handles any resolution / scope).
isfile(GRID_CELLS_CSV) ||
    error("$GRID_CELLS_CSV not found. Run data_prep/build_bay_area_grid.py.")
grid_meta = CSV.read(GRID_CELLS_CSV, DataFrame)
lon_min, lon_max = extrema(grid_meta.lon_lo)
lat_min, lat_max = extrema(grid_meta.lat_lo)
# Extend by one cell on the upper edge so the last column/row isn't clipped.
cell_dlon = (grid_meta.lon_hi[1] - grid_meta.lon_lo[1])
cell_dlat = (grid_meta.lat_hi[1] - grid_meta.lat_lo[1])
lon_max += cell_dlon
lat_max += cell_dlat
region = [lon_min, lon_max, lat_min, lat_max]
println("  region (lon_min, lon_max, lat_min, lat_max) = $region")

# ============================================================================
# Build a dense grid matrix from sparse (cell_i, cell_j, value) records
# ============================================================================
"""
Project the cell-keyed values into a regular grid suitable for `grdimage`.
Cells outside the in-mask set (water / out-of-region) become NaN.
"""
function build_grid(cell_df::DataFrame, value_col::Symbol, grid_meta::DataFrame)
    imax = maximum(grid_meta.cell_i)
    jmax = maximum(grid_meta.cell_j)
    M = fill(NaN, jmax, imax)   # row = j (lat), col = i (lon)
    for row in eachrow(cell_df)
        M[row.cell_j, row.cell_i] = row[value_col]
    end
    return M
end

# Coordinate vectors aligned with the dense matrix.
imax = maximum(grid_meta.cell_i)
jmax = maximum(grid_meta.cell_j)
X = collect(lon_min .+ ((1:imax) .- 0.5) .* cell_dlon)
Y = collect(lat_min .+ ((1:jmax) .- 0.5) .* cell_dlat)

# ============================================================================
# Per-policy outcome maps (sequential colormap; range = observed reward range)
# ============================================================================
"""
Render one policy's outcome heatmap. `value_col` picks the column.
"""
function plot_outcome(cell_df, value_col::Symbol, label::String, out_name::String)
    M = build_grid(cell_df, value_col, grid_meta)
    vals = filter(!isnan, vec(M))
    lo = round(quantile(vals, 0.02); digits = 3)
    hi = round(quantile(vals, 0.98); digits = 3)
    cpt = makecpt(color = :hot, range = (lo, hi, (hi - lo) / 200))

    out_path = joinpath(OUT_DIR, out_name)
    # mat2grid + grdimage is robust to the pcolor edge/centre bookkeeping in
    # GMT.jl (pcolor threw BoundsError on a 198-column grid). Rows of M run
    # south to north, matching GMT.jl's default "BCB" layout. NaN cells
    # (water, outside the mask) are left transparent.
    G = mat2grid(M; x = X, y = Y)
    grdimage(G;
        proj  = "M6i",
        cmap  = cpt,
        region = region,
        frame = "afg",
        title = label,
        nan_alpha = true)
    coast!(region = region, proj = "M6i", area = 1000, water = :white)
    coast!(region = region, proj = "M6i", shorelines = true)
    annotate_cities!()
    colorbar!(cmap = cpt, show = false, savefig = out_path)
    println("✓ $out_path")
end

# ============================================================================
# Pairwise difference maps (diverging colormap, symmetric around zero — no clipping)
# ============================================================================
function plot_difference(cell_df, value_col::Symbol, label::String, out_name::String)
    M = build_grid(cell_df, value_col, grid_meta)
    vals = filter(!isnan, vec(M))
    isempty(vals) && (println("  Skipping $out_name: no data"); return)

    # Symmetric range so the diverging colormap centers on zero. Use a robust
    # quantile (98th of |Δ|) so a couple of outlier cells don't squash the
    # rest of the map.
    half = round(quantile(abs.(vals), 0.98); digits = 4)
    half = max(half, 0.001)   # avoid degenerate colormap

    cpt = makecpt(color = :polar, range = (-half, half, half / 100))

    out_path = joinpath(OUT_DIR, out_name)
    # mat2grid + grdimage is robust to the pcolor edge/centre bookkeeping in
    # GMT.jl (pcolor threw BoundsError on a 198-column grid). Rows of M run
    # south to north, matching GMT.jl's default "BCB" layout. NaN cells
    # (water, outside the mask) are left transparent.
    G = mat2grid(M; x = X, y = Y)
    grdimage(G;
        proj  = "M6i",
        cmap  = cpt,
        region = region,
        frame = "afg",
        title = label,
        nan_alpha = true)
    coast!(region = region, proj = "M6i", area = 1000, water = :white)
    coast!(region = region, proj = "M6i", shorelines = true)
    annotate_cities!()
    colorbar!(cmap = cpt, show = false, savefig = out_path)
    n_neg = count(<(0), vals)
    pct_neg = round(100 * n_neg / length(vals); digits = 1)
    println("✓ $out_path  ($n_neg cells negative — $pct_neg %)")
end

# ============================================================================
# City annotation overlay (shared by every map)
# ============================================================================
function annotate_cities!()
    in_region = filter(c -> region[1] <= c[1] <= region[2] &&
                            region[3] <= c[2] <= region[4],
                       CITIES)
    isempty(in_region) && return

    lons  = [c[1] for c in in_region]
    lats  = [c[2] for c in in_region]
    names = [c[3] for c in in_region]

    plot!(lons, lats;
        symbol = "c0.12c",
        fill   = :black,
        region = region,
        proj   = "M6i",
        show   = false)
    for (lon, lat, name) in in_region
        text!(name;
            x = lon, y = lat,
            region = region, proj = "M6i",
            font = "7p,Helvetica,black",
            fill = :white,
            pen  = "0.3p,black",
            offset = (shift = (0, 0.25),),
            show = false)
    end
end

# ============================================================================
# Main — render every figure
# ============================================================================
mkpath(OUT_DIR)

println("\n[1/2] Per-policy outcome maps")
plot_outcome(cell, :reward_optimal, "MDP-optimal policy: P(good outcome)",
             "CA_outcome_optimal.pdf")
plot_outcome(cell, :reward_nearest, "Nearest-hospital baseline: P(good outcome)",
             "CA_outcome_nearest.pdf")
plot_outcome(cell, :reward_heur1, "Heuristic 1: P(good outcome)",
             "CA_outcome_heur1.pdf")
plot_outcome(cell, :reward_heur2, "Heuristic 2: P(good outcome)",
             "CA_outcome_heur2.pdf")

println("\n[2/2] Policy-difference maps (diverging cmap, no clipping)")
plot_difference(cell, :diff_opt_minus_nearest,
                "MDP optimal minus Nearest", "CA_diff_opt_minus_nearest.pdf")
plot_difference(cell, :diff_opt_minus_heur1,
                "MDP optimal minus Heuristic 1", "CA_diff_opt_minus_heur1.pdf")
plot_difference(cell, :diff_opt_minus_heur2,
                "MDP optimal minus Heuristic 2", "CA_diff_opt_minus_heur2.pdf")
plot_difference(cell, :diff_heur1_minus_nearest,
                "Heuristic 1 minus Nearest", "CA_diff_heur1_minus_nearest.pdf")
plot_difference(cell, :diff_heur2_minus_nearest,
                "Heuristic 2 minus Nearest", "CA_diff_heur2_minus_nearest.pdf")

println("\nDone.")
