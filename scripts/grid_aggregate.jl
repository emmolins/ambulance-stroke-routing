# =============================================================================
# scripts/grid_aggregate.jl
#
# Pool the per-cell grid CSVs produced by `grid_generator.jl` into a single
# wide CSV that downstream plotting/analysis can consume.
#
# Inputs
#   sampled_points/grid_plot_csvs_CA_v2/cell_<i>_<j>.csv  (per-cell, N_PER_CELL
#                                                          patient rows each)
#
# Outputs
#   sampled_points/CA_grid_patients.csv     — pooled per-patient rows
#                                              (cell_i, cell_j, patient state,
#                                               all 4 policies' action/reward/tt)
#   sampled_points/CA_grid_cell_means.csv   — one row per cell, mean reward for
#                                              each policy + paired Δ columns
#
# The cell-means file is what feeds the heatmaps. Per-patient rows are kept so
# any later analysis (e.g. by stroke-type, by distance, by t_onset bin) can
# be done on the same data.
# =============================================================================

using CSV
using DataFrames
using Statistics

const IN_DIR        = "sampled_points/grid_plot_csvs_CA_v2"
const PATIENTS_CSV  = "sampled_points/CA_grid_patients.csv"
const CELL_MEAN_CSV = "sampled_points/CA_grid_cell_means.csv"

isdir(IN_DIR) || error("Generator output not found at $IN_DIR. " *
                        "Run scripts/grid_generator.jl first.")

# ---------------------------------------------------------------------------
# 1. Pool per-patient rows
# ---------------------------------------------------------------------------
println("Reading per-cell CSVs from $IN_DIR ...")
files = filter(f -> startswith(f, "cell_") && endswith(f, ".csv"),
               readdir(IN_DIR))
isempty(files) && error("No cell_*.csv files found.")

pooled = DataFrame()
n_empty = 0
for f in files
    path = joinpath(IN_DIR, f)
    # Empty sentinel files (zero bytes) mark cells where the probe said
    # routable but all sample attempts dropped; skip them.
    if filesize(path) == 0
        n_empty += 1
        continue
    end
    df = CSV.read(path, DataFrame)
    pooled = isempty(pooled) ? df : vcat(pooled, df; cols = :union)
end

println("  Cells with data : $(length(files) - n_empty)")
println("  Empty sentinels : $n_empty")
println("  Pooled rows     : $(nrow(pooled))")

mkpath(dirname(PATIENTS_CSV))
CSV.write(PATIENTS_CSV, pooled)
println("  ✓ Wrote $PATIENTS_CSV")
println()

# ---------------------------------------------------------------------------
# 2. Cell-level summary (mean per policy + paired differences)
# ---------------------------------------------------------------------------
println("Computing per-cell means and pairwise differences ...")
cell_mean = combine(
    groupby(pooled, [:cell_i, :cell_j]),
    :start_lat              => mean => :cell_lat,
    :start_lon              => mean => :cell_lon,
    :optimal_action_reward   => mean => :reward_optimal,
    :nearest_hospital_reward => mean => :reward_nearest,
    :heuristic_1_reward      => mean => :reward_heur1,
    :heuristic_2_reward      => mean => :reward_heur2,
    nrow                     => :n_samples,
)

# Paired contrasts (NO clipping — negative values are real and informative).
cell_mean.diff_opt_minus_nearest = cell_mean.reward_optimal .- cell_mean.reward_nearest
cell_mean.diff_opt_minus_heur1   = cell_mean.reward_optimal .- cell_mean.reward_heur1
cell_mean.diff_opt_minus_heur2   = cell_mean.reward_optimal .- cell_mean.reward_heur2
cell_mean.diff_heur1_minus_nearest = cell_mean.reward_heur1 .- cell_mean.reward_nearest
cell_mean.diff_heur2_minus_nearest = cell_mean.reward_heur2 .- cell_mean.reward_nearest

CSV.write(CELL_MEAN_CSV, cell_mean)
println("  ✓ Wrote $CELL_MEAN_CSV  ($(nrow(cell_mean)) cells × $(ncol(cell_mean)) cols)")
println()

# ---------------------------------------------------------------------------
# 3. Quick sanity summary
# ---------------------------------------------------------------------------
function summarize_diff(label, v)
    n_neg = count(<(0), v)
    n_pos = count(>(0), v)
    println("  $label")
    println("    cells with Δ < 0 : $n_neg ($(round(100 * n_neg / length(v), digits=1))%)")
    println("    cells with Δ > 0 : $n_pos ($(round(100 * n_pos / length(v), digits=1))%)")
    println("    median Δ         : $(round(median(v), digits=5))")
    println("    range            : [$(round(minimum(v), digits=5)), $(round(maximum(v), digits=5))]")
end

println("Per-cell paired-difference summary (REAL values, no clipping):")
summarize_diff("MDP optimal − Nearest", cell_mean.diff_opt_minus_nearest)
summarize_diff("MDP optimal − Heuristic 1", cell_mean.diff_opt_minus_heur1)
summarize_diff("Heuristic 1 − Nearest",    cell_mean.diff_heur1_minus_nearest)
println()
println("These are the inputs the heatmap plotter consumes — without clipping.")
