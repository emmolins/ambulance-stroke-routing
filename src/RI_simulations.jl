# =============================================================================
# RI_simulations.jl
# Population-weighted patient simulation for the Rhode Island generalizability
# analyses of the ambulance stroke-routing study.
#
# This script is a structural parallel to CA_simulations.jl. The MDP, policies,
# and analysis pipeline are identical; only the patient-location pool and the
# included MDP module differ (RI_STPMDP_ORS.jl uses Rhode Island hospitals and
# census-tract-derived patient locations).
#
# For each sampled patient, four routing policies are evaluated under a paired
# comparison:
#
#   1. MDP-OPTIMAL      best_action with information-aware forward search
#                       (marginalizes the post-arrival diagnosis over the
#                       stroke-type prior; see RI_STPMDP_ORS.jl)
#   2. NEAREST          current practice: nearest hospital of any type
#   3. HEURISTIC_1      nearest Comprehensive Stroke Center, with fallback
#                       chain (PSC, then any hospital)
#   4. HEURISTIC_2      nearest Primary OR Comprehensive Stroke Center, with
#                       fallback to any hospital
#
# Outputs
#   simulation_results/RI_simulation_results[_rep<N>][_noise<P>].csv
#       One row per successful patient (16 columns).
#   simulation_results/RI_simulation_dropouts[_rep<N>][_noise<P>].csv
#       One row per patient that could not be evaluated.
#
# Replication protocol (IEEE multi-seed convention)
#   Pass REPLICATE_INDEX as an environment variable to run independent
#   replicates with distinct random seeds:
#       REPLICATE_INDEX=1 julia --project=. RI_simulations.jl
#   Each replicate shuffles the canonical patient pool with the replicate
#   seed and draws N_SIMULATIONS patients without replacement.
#
# IMPORTANT: ORS routing graph
#   This script makes ORS calls against http://localhost:8080/ors/v2/... and
#   expects the ORS instance to have a Rhode Island OSM graph loaded. If your
#   ORS container is still on a California graph (as during the main CA runs),
#   reload it with the RI PBF before running. See ORS_SETUP.md.
# =============================================================================

using CSV
using DataFrames
using HTTP
using ProgressMeter
using Random
using StatsBase

include("RI_STPMDP_ORS.jl")

# -----------------------------------------------------------------------------
# Configuration (globals because RI_STPMDP_ORS.jl pre-declares SEED non-const)
# -----------------------------------------------------------------------------

# Replication
BASE_SEED       = 1234
REPLICATE_INDEX = haskey(ENV, "REPLICATE_INDEX") ? parse(Int, ENV["REPLICATE_INDEX"]) : 0
SEED            = REPLICATE_INDEX == 0 ? BASE_SEED : BASE_SEED + 10_000 * REPLICATE_INDEX

# Run parameters
N_SIMULATIONS = 1000           # patients per replicate; 10 replicates → 10,000 total
MDP_DEPTH     = 2
POINTS_CSV    = "sampled_points/RI_points.csv"

# Travel-time sensitivity is applied POST-HOC (see scripts/recompute_perturbed_rewards.jl).
# Simulations always run against deterministic ORS — the planner sees only
# what the dispatcher could see at t = 0.

# Output paths (auto-tagged with replicate index)
_rep_tag      = REPLICATE_INDEX == 0 ? "" : "_rep$(REPLICATE_INDEX)"
RESULTS_CSV   = "simulation_results/RI_simulation_results$(_rep_tag).csv"
DROPOUTS_CSV  = "simulation_results/RI_simulation_dropouts$(_rep_tag).csv"

# Ordered policy spec: (name, policy function returning an Action or nothing)
const POLICIES = (
    ("optimal", (mdp, s) -> best_action(mdp, s, MDP_DEPTH)),
    ("nearest", (mdp, s) -> let a = current_practice_action(mdp, s);
                              a === nothing ? nothing : string_to_enum(a)
                            end),
    ("heur1",   (mdp, s) -> let a = heuristic_1_action(mdp, s);
                              a === nothing ? nothing : string_to_enum(a)
                            end),
    ("heur2",   (mdp, s) -> let a = heuristic_2_action(mdp, s);
                              a === nothing ? nothing : string_to_enum(a)
                            end),
)

# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------

struct SimulationResult
    state        :: PatientState
    actions      :: NTuple{4, String}    # in POLICIES order
    rewards      :: NTuple{4, Float64}
    travel_times :: NTuple{4, Float64}
end

struct Dropout
    sim_attempt   :: Int
    start_lat     :: Float64
    start_lon     :: Float64
    t_onset       :: Float64
    stroke_type   :: String
    failed_policy :: String
    reason        :: String
end

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

"""
Pop one patient location off the front of a pre-shuffled pool (without
replacement). `main()` shuffles the canonical pool with the replicate-specific
seed before passing it in; that gives each replicate a deterministic but
distinct ordering and lets us draw 1,000 unique patients per replicate from
the 10,000-point canonical pool.
"""
function sample_location!(pool::Vector{Tuple{Float64, Float64}})
    isempty(pool) && error("Pool exhausted: increase the canonical sample size " *
                           "or decrease N_SIMULATIONS.")
    return popfirst!(pool)
end

"""
Evaluate one policy on one patient.

Returns `((action_str, reward, travel_time), nothing)` on success or
`(nothing, reason_string)` on a recoverable failure. Unexpected exceptions
propagate so real bugs surface.
"""
function evaluate_policy(mdp, state, policy_fn)
    action = policy_fn(mdp, state)
    action === nothing && return (nothing, "no valid action")

    action_str = enum_to_string(action)
    try
        next_state = rand(transition(mdp, state, action))
        r          = reward(mdp, state, action, next_state)
        tt         = calculate_travel_time(state.loc, next_state.loc)
        tt === nothing && return (nothing, "ORS could not route")
        return ((action_str, r, tt), nothing)
    catch e
        if isa(e, HTTP.Exceptions.StatusError) && e.status == 404
            return (nothing, "ORS 404")
        end
        rethrow(e)
    end
end

# -----------------------------------------------------------------------------
# Simulation loop
# -----------------------------------------------------------------------------

"""
Run N_SIMULATIONS patients through all four policies.

Paired-comparison design: a patient is included in the results CSV only if
ALL four policies returned a valid evaluation. Patients where any policy
failed are recorded in `dropouts` for downstream selection-bias analysis.
"""
function run_simulation(mdp, location_pool)
    results  = SimulationResult[]
    dropouts = Dropout[]
    progress = Progress(N_SIMULATIONS; desc = "RI sims ")
    attempts = 0

    while length(results) < N_SIMULATIONS
        attempts += 1
        latlon = sample_location!(location_pool)

        state = PatientState(
            Location("FIELD$attempts", latlon, -1, FIELD),
            30 + rand() * 240,
            UNKNOWN,
            sample_stroke_type(mdp),
        )

        # Evaluate every policy; bail at first failure to preserve pairing.
        outputs = Vector{Tuple{String, Float64, Float64}}(undef, length(POLICIES))
        failure = nothing
        for (i, (name, policy_fn)) in enumerate(POLICIES)
            (out, reason) = evaluate_policy(mdp, state, policy_fn)
            if out === nothing
                failure = (name, reason)
                break
            end
            outputs[i] = out
        end

        if failure !== nothing
            push!(dropouts, Dropout(
                attempts, latlon[1], latlon[2],
                state.t_onset, string(state.stroke_type),
                failure[1], failure[2]))
            continue
        end

        push!(results, SimulationResult(
            state,
            ntuple(i -> outputs[i][1], length(POLICIES)),
            ntuple(i -> outputs[i][2], length(POLICIES)),
            ntuple(i -> outputs[i][3], length(POLICIES)),
        ))
        next!(progress)
    end
    finish!(progress)

    return results, dropouts, attempts
end

# -----------------------------------------------------------------------------
# Output formatting
# -----------------------------------------------------------------------------

# Column-name suffixes match the legacy schema used by RI_simulations_stats.jl.
const RESULT_COL_SUFFIX = (
    optimal = ("optimal_action",          "optimal_action_reward",   "travel_time_optimal"),
    nearest = ("nearest_hospital_action", "nearest_hospital_reward", "travel_time_nh"),
    heur1   = ("heuristic_1_action",      "heuristic_1_reward",      "travel_time_h1"),
    heur2   = ("heuristic_2_action",      "heuristic_2_reward",      "travel_time_h2"),
)

function results_to_dataframe(results)
    df = DataFrame(
        start_lat   = [r.state.loc.latlon[1]      for r in results],
        start_lon   = [r.state.loc.latlon[2]      for r in results],
        t_onset     = [r.state.t_onset            for r in results],
        stroke_type = [string(r.state.stroke_type) for r in results],
    )
    for (i, (name, _)) in enumerate(POLICIES)
        a_col, r_col, t_col = RESULT_COL_SUFFIX[Symbol(name)]
        df[!, a_col] = [r.actions[i]      for r in results]
        df[!, r_col] = [r.rewards[i]      for r in results]
        df[!, t_col] = [r.travel_times[i] for r in results]
    end
    return df
end

function dropouts_to_dataframe(dropouts)
    return DataFrame(
        sim_attempt   = getfield.(dropouts, :sim_attempt),
        start_lat     = getfield.(dropouts, :start_lat),
        start_lon     = getfield.(dropouts, :start_lon),
        t_onset       = getfield.(dropouts, :t_onset),
        stroke_type   = getfield.(dropouts, :stroke_type),
        failed_policy = getfield.(dropouts, :failed_policy),
        reason        = getfield.(dropouts, :reason),
    )
end

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

function main()
    Random.seed!(SEED)
    println("Replicate $(REPLICATE_INDEX)  |  seed $(SEED)")

    println("Loading location pool from $POINTS_CSV ...")
    pool_raw = [(row.Latitude, row.Longitude) for row in CSV.File(POINTS_CSV)]
    println("  $(length(pool_raw)) candidate locations loaded")

    # Per-replicate shuffle: deterministic given SEED, distinct across replicates.
    pool = shuffle(pool_raw)
    if N_SIMULATIONS > length(pool)
        error("N_SIMULATIONS=$N_SIMULATIONS exceeds pool size $(length(pool)); " *
              "regenerate RI_points.csv with more samples or lower N_SIMULATIONS.")
    end

    mdp = StrokeMDP()
    println("Running $N_SIMULATIONS simulations (depth=$MDP_DEPTH) ...")

    results, dropouts, attempts = run_simulation(mdp, pool)

    mkpath(dirname(RESULTS_CSV))
    CSV.write(RESULTS_CSV,  results_to_dataframe(results))
    CSV.write(DROPOUTS_CSV, dropouts_to_dataframe(dropouts))

    print_summary(results, dropouts, attempts)
    return results, dropouts
end

function print_summary(results, dropouts, attempts)
    n_dropped = length(dropouts)
    drop_pct  = round(100 * n_dropped / max(attempts, 1), digits = 1)

    println()
    println("┌─ Simulation summary ─────────────────────────────────────")
    println("│  Successful patients:  $(length(results))")
    println("│  Dropped patients:     $n_dropped  ($drop_pct% of $attempts attempts)")
    if n_dropped > 0
        for (key, label) in ((:failed_policy, "failing policy"), (:reason, "reason"))
            counts = countmap(getproperty.(dropouts, key))
            println("│  Dropouts by $label:")
            for (k, v) in sort(collect(counts), by = x -> -x[2])
                println("│    $k: $v")
            end
        end
    end
    println("└──────────────────────────────────────────────────────────")
    println("Wrote $RESULTS_CSV")
    println("Wrote $DROPOUTS_CSV")
end

main()
