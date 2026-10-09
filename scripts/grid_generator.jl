# =============================================================================
# scripts/grid_generator.jl
#
# Generate per-cell patient simulations for the Bay Area outcome heatmap.
# Mirrors the per-patient pipeline used in src/CA_simulations.jl (same MDP,
# same patient distribution, same four policies) so the heatmap is
# methodologically consistent with the main results.
#
# Geographic scope:
#   The Bay Area is defined by a precomputed cell mask
#   `sampled_points/bay_area_grid_cells.csv`, produced by
#   `data_prep/build_bay_area_grid.py` from the 187-ZIP-code shapefile
#   `data_prep/data/bayarea_zipcodes.shp`. The Python helper dissolves the
#   ZIPs into a single Bay Area polygon, lays a regular grid over its WGS84
#   bbox, and keeps only cells whose centre is inside the polygon.
#
#   This script does NOT do any geospatial work itself — it just iterates
#   over the cells the Python helper produced. To change resolution or
#   region definition, edit and re-run the Python helper.
#
# Sample size: N_PER_CELL patients per cell (default 20).
# Reward semantics: REALIZED outcome (planner sees UNKNOWN; reward computed
#                   under KNOWN/true stroke type).
#
# Output:
#   sampled_points/grid_plot_csvs_CA_v2/cell_<i>_<j>.csv  (one per cell)
#
# Resumability:
#   Cells with an existing output CSV are skipped. Kill and restart any time.
#
# Reproducibility:
#   RNG is seeded as a function of (cell_i, cell_j); rerunning a single cell
#   yields identical samples.
#
# Run:
#   # Prereq: ORS must be running with the California road graph.
#   #         Cell mask must exist (build_bay_area_grid.py).
#   julia --project=. scripts/grid_generator.jl
#
# Smoke test (fast):
#   N_PER_CELL=3 CELL_LIMIT=50 julia --project=. scripts/grid_generator.jl
# =============================================================================

using CSV
using DataFrames
using Random
using ProgressMeter
using HTTP

include("../src/CA_STPMDP_ORS.jl")

# ----- Configuration ---------------------------------------------------------
const CELLS_CSV   = "sampled_points/bay_area_grid_cells.csv"
const OUT_DIR     = get(ENV, "OUT_DIR", "sampled_points/grid_plot_csvs_CA_v2")
const N_PER_CELL  = haskey(ENV, "N_PER_CELL") ? parse(Int, ENV["N_PER_CELL"]) : 20
const CELL_LIMIT  = haskey(ENV, "CELL_LIMIT") ? parse(Int, ENV["CELL_LIMIT"]) : typemax(Int)
const BASE_SEED   = 7777
const MDP_DEPTH   = 2
const MAX_ATTEMPTS_PER_CELL = 5 * N_PER_CELL   # cap dropouts

# ----- Policy registry — matches src/CA_simulations.jl ---------------------
policy_optimal(mdp, state)  = best_action(mdp, state, MDP_DEPTH)
function policy_nearest(mdp, state)
    s = current_practice_action(mdp, state)
    return s === nothing ? nothing : string_to_enum(s)
end
function policy_heur1(mdp, state)
    s = heuristic_1_action(mdp, state)
    return s === nothing ? nothing : string_to_enum(s)
end
function policy_heur2(mdp, state)
    s = heuristic_2_action(mdp, state)
    return s === nothing ? nothing : string_to_enum(s)
end

const POLICIES = (policy_optimal, policy_nearest, policy_heur1, policy_heur2)

cell_seed(i, j) = BASE_SEED + 1_000_000 * i + j

# ----- Per-policy evaluation under REALIZED rewards -------------------------
"""
The planner sees UNKNOWN diagnosis (correct decision-time information). The
reward is then *recomputed* under the patient's true stroke type (KNOWN
branch), matching the realized-outcome semantics used in the per-replicate
pipeline.
"""
function evaluate_policy_realized(mdp, planner_state, realized_type, policy_fn)
    action = policy_fn(mdp, planner_state)
    action === nothing && return (nothing, "no valid action")

    action_str = enum_to_string(action)
    try
        s_known = PatientState(planner_state.loc, planner_state.t_onset,
                               KNOWN, realized_type)
        next_state = rand(transition(mdp, s_known, action))
        r = action_value(mdp, s_known, action, next_state)
        # Same value as calculate_travel_time, but served from the cached matrix row
        # (avoids 4 per-pair Directions calls per patient).
        tt = cached_travel_time(mdp, planner_state.loc, next_state.loc)
        tt === nothing && return (nothing, "ORS could not route")
        return ((action_str, r, tt), nothing)
    catch e
        if isa(e, HTTP.Exceptions.StatusError) && e.status == 404
            return (nothing, "ORS 404")
        else
            rethrow(e)
        end
    end
end

# ----- Sample N_PER_CELL patients for a single cell -------------------------
function process_cell(mdp, cell)
    Random.seed!(cell_seed(cell.cell_i, cell.cell_j))

    lat_lo, lat_hi = cell.lat_lo, cell.lat_hi
    lon_lo, lon_hi = cell.lon_lo, cell.lon_hi

    rows = NamedTuple[]
    attempts = 0

    while length(rows) < N_PER_CELL && attempts < MAX_ATTEMPTS_PER_CELL
        attempts += 1

        # Uniform random location within the cell.
        lat = lat_lo + rand() * (lat_hi - lat_lo)
        lon = lon_lo + rand() * (lon_hi - lon_lo)

        loc = Location("FIELD$(cell.cell_i)_$(cell.cell_j)_$(attempts)",
                       (lat, lon), -1, FIELD)
        t_onset = sample_onset_time()
        realized_type = sample_stroke_type(mdp)
        planner_state = PatientState(loc, t_onset, UNKNOWN, realized_type)

        # Evaluate all four policies. Require all to succeed (paired design).
        policy_outputs = Vector{Tuple{String, Float64, Float64}}(undef, 4)
        ok = true
        for (k, policy_fn) in enumerate(POLICIES)
            (result, _) = evaluate_policy_realized(mdp, planner_state,
                                                    realized_type, policy_fn)
            if result === nothing
                ok = false
                break
            end
            policy_outputs[k] = result
        end
        ok || continue

        push!(rows, (
            cell_i        = cell.cell_i,
            cell_j        = cell.cell_j,
            start_lat     = lat,
            start_lon     = lon,
            t_onset       = t_onset,
            stroke_type   = string(realized_type),
            optimal_action          = policy_outputs[1][1],
            optimal_action_reward   = policy_outputs[1][2],
            travel_time_optimal     = policy_outputs[1][3],
            nearest_hospital_action = policy_outputs[2][1],
            nearest_hospital_reward = policy_outputs[2][2],
            travel_time_nh          = policy_outputs[2][3],
            heuristic_1_action      = policy_outputs[3][1],
            heuristic_1_reward      = policy_outputs[3][2],
            travel_time_h1          = policy_outputs[3][3],
            heuristic_2_action      = policy_outputs[4][1],
            heuristic_2_reward      = policy_outputs[4][2],
            travel_time_h2          = policy_outputs[4][3],
        ))
    end

    return DataFrame(rows), attempts
end

# ============================================================================
# Main
# ============================================================================
isfile(CELLS_CSV) ||
    error("Cell mask not found at $CELLS_CSV.\n" *
          "Run data_prep/build_bay_area_grid.py first " *
          "(from data_prep/ with the venv active).")

println("="^78)
println("BAY AREA GRID GENERATOR")
println("="^78)
cells = CSV.read(CELLS_CSV, DataFrame)
println("  Cell mask          : $CELLS_CSV ($(nrow(cells)) cells)")
println("  Samples per cell   : $N_PER_CELL")
println("  Output directory   : $OUT_DIR")
println("  Reward semantics   : realized (KNOWN diagnosis at evaluation)")
println("  MDP depth          : $MDP_DEPTH")
CELL_LIMIT < typemax(Int) && println("  Cell limit (smoke) : $CELL_LIMIT")
println()

mkpath(OUT_DIR)

println("Building MDP (loading hospitals)...")
mdp = StrokeMDP()
println("  $(length(mdp.locations)) locations loaded")
println()

n_to_process = min(nrow(cells), CELL_LIMIT)
progress = Progress(n_to_process; desc = "grid ")
written = 0
skipped_existing = 0
skipped_dropped = 0

for r in 1:n_to_process
    global written, skipped_existing, skipped_dropped
    cell = cells[r, :]
    out_path = joinpath(OUT_DIR, "cell_$(cell.cell_i)_$(cell.cell_j).csv")

    if isfile(out_path)
        skipped_existing += 1
        next!(progress)
        continue
    end

    df, attempts = process_cell(mdp, cell)
    if nrow(df) == 0
        # All sample attempts failed (likely unroutable cluster).
        # Write empty sentinel so we don't retry on resume.
        touch(out_path)
        skipped_dropped += 1
    else
        CSV.write(out_path, df)
        written += 1
    end
    next!(progress)
end

finish!(progress)

println()
println("="^78)
println("Cells written this run : $written")
println("Cells skipped (exists) : $skipped_existing")
println("Cells dropped (ORS fail): $skipped_dropped")
println()
println("Next:")
println("  julia --project=. scripts/grid_aggregate.jl   # pool cells into one CSV")
println("  julia --project=. viz/CA_grid_maker.jl        # render the heatmaps")
