#=
File: CA_simulations.jl
================================================================================
Population-weighted patient simulation for the California analyses of the
ambulance stroke-routing study.

For each sampled patient, four routing policies are evaluated head-to-head:

  1. MDP-OPTIMAL    — best_action with information-aware forward search
                      (marginalizes the post-arrival reveal over the stroke-type
                      prior; see CA_STPMDP_ORS.jl for details)
  2. NEAREST        — current CA practice: route to the nearest hospital of any
                      type
  3. HEURISTIC 1    — route to the nearest Comprehensive Stroke Center (CSC)
  4. HEURISTIC 2    — route to the nearest Primary OR Comprehensive Stroke Center

OUTPUTS
-------
  simulation_results/CA_simulation_results.csv
      One row per successful simulation. 16 columns:
        start_lat, start_lon, t_onset, stroke_type,
        and {action, reward, travel_time} for each of the 4 policies.

  simulation_results/CA_simulation_dropouts.csv
      One row per patient that could not be evaluated (because at least one of
      the four policies returned no valid action, or an ORS routing call failed).
      Columns identify the failing policy and the reason — this lets the analysis
      step assess potential selection bias rather than burying it.

SAMPLE-SIZE JUSTIFICATION
-------------------------
  Empirically (from a pilot run), the per-patient reward standard deviation
  ranges from ~0.02 (within hemorrhagic patients) to ~0.10 (across the full
  patient population). For a paired comparison between two policies on the same
  N patients, the standard error of the mean difference is sd_diff / sqrt(N),
  where sd_diff is typically smaller than the marginal sd because of within-
  patient correlation.

  At N = 1000, the 95% CI half-width on any pairwise mean-reward difference is
  ~1.96 * 0.05 / sqrt(1000) ≈ 0.003, well below the smallest clinically
  meaningful effect size (Δ ≈ 0.01 in P(good outcome)).

  The smoke-test default below is N_SIMULATIONS = 10; switch to 1000 for paper
  runs.

REPRODUCIBILITY
---------------
  A single global RNG seed (SEED) controls patient location order, onset-time
  draws, stroke-type draws, and any MDP-internal randomness. The location pool
  is shuffled deterministically from CA_points.csv before sampling.

  To produce bootstrap confidence intervals on policy differences without
  rerunning the simulation, the downstream CA_simulations_stats.jl script
  bootstrap-resamples the per-patient paired-difference vector.
================================================================================
=#

using CSV
using DataFrames
using StatsBase
using Random
using ProgressMeter
using HTTP

include("CA_STPMDP_ORS.jl")

# ============================================================================
# Configuration
# ============================================================================
# Note: these are plain globals (not `const`) because CA_STPMDP_ORS.jl already
# declares SEED as a non-const global at load time, and Julia disallows
# upgrading an existing global into a const within the same module.

# ----- Replication structure (IEEE-style multi-seed protocol) ---------------
# Set REPLICATE_INDEX via env var when running multiple replicates in parallel:
#     REPLICATE_INDEX=1 julia --project=. CA_simulations.jl
#     REPLICATE_INDEX=2 julia --project=. CA_simulations.jl   # in another shell
# Each replicate uses a different effective seed and writes to its own
# output CSV (CA_simulation_results_rep<N>.csv).
BASE_SEED       = 1234
REPLICATE_INDEX = haskey(ENV, "REPLICATE_INDEX") ? parse(Int, ENV["REPLICATE_INDEX"]) : 0
SEED            = REPLICATE_INDEX == 0 ? BASE_SEED : BASE_SEED + 10_000 * REPLICATE_INDEX

# ----- Run parameters --------------------------------------------------------
N_SIMULATIONS = 1000              # patients per replicate; 10 replicates → 10,000 total
MDP_DEPTH     = 2                 # Forward-search horizon for MDP-optimal policy
POINTS_CSV    = "sampled_points/CA_points.csv"

# Travel-time sensitivity (variance / systematic bias) is applied POST-HOC by
# scripts/recompute_perturbed_rewards.jl. The simulation always runs against
# deterministic ORS travel times — exactly what the dispatcher would see in
# practice at t = 0.

# ----- Output paths (auto-tagged with replicate index) ----------------------
_rep_tag      = REPLICATE_INDEX == 0 ? "" : "_rep$(REPLICATE_INDEX)"
# Optional OUTPUT_TAG (e.g. "_evtonsite") keeps sensitivity runs apart from the baseline files.
_out_tag      = get(ENV, "OUTPUT_TAG", "")
RESULTS_CSV   = "simulation_results/CA_simulation_results$(_rep_tag)$(_out_tag).csv"
DROPOUTS_CSV  = "simulation_results/CA_simulation_dropouts$(_rep_tag)$(_out_tag).csv"

POLICY_NAMES  = ("optimal", "nearest", "heur1", "heur2")

Random.seed!(SEED)

println("Replicate index : $(REPLICATE_INDEX) (effective seed = $SEED)")

# ============================================================================
# Types
# ============================================================================
struct SimulationResult
    start_state  :: PatientState
    actions      :: NTuple{4, String}    # in POLICY_NAMES order
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

# ============================================================================
# Helpers
# ============================================================================
# Without-replacement sampling: pop the next patient off a pre-shuffled pool.
# Each replicate shuffles the canonical 10,000-point pool with its own seed
# before passing it in (see `main()` below), so:
#   * Within a replicate, every patient location is unique (N=1000 << 10,000).
#   * Across replicates, the per-replicate shuffles use different RNG state and
#     therefore yield different (overlapping) 1,000-patient subsets.
function sample_location!(pool::Vector{Tuple{Float64, Float64}})
    isempty(pool) && error("Pool exhausted: increase canonical sample size " *
                           "or decrease N_SIMULATIONS.")
    return popfirst!(pool)
end

# Each policy fn: (mdp, state) -> Action enum OR nothing on failure
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

POLICIES = (policy_optimal, policy_nearest, policy_heur1, policy_heur2)

# Evaluate one policy on one patient.
# Returns either (action_str, reward, travel_time) on success, or
# (nothing, reason_string) on failure.
function evaluate_policy(mdp, state, policy_fn)
    action = policy_fn(mdp, state)
    action === nothing && return (nothing, "no valid action")

    action_str = enum_to_string(action)
    try
        next_state = rand(transition(mdp, state, action))
        r = action_value(mdp, state, action, next_state)
        tt = calculate_travel_time(state.loc, next_state.loc)
        tt === nothing && return (nothing, "ORS could not route")
        return ((action_str, r, tt), nothing)
    catch e
        # Only swallow expected exception classes (ORS 404 routing failures);
        # let everything else bubble up so real bugs are surfaced.
        if isa(e, HTTP.Exceptions.StatusError) && e.status == 404
            return (nothing, "ORS 404")
        else
            rethrow(e)
        end
    end
end

# ============================================================================
# Simulation loop
# ============================================================================
function run_simulation(mdp, location_pool)
    results  = SimulationResult[]
    dropouts = Dropout[]
    progress = Progress(N_SIMULATIONS; desc = "CA sims ")
    attempts = 0

    while length(results) < N_SIMULATIONS
        attempts += 1
        latlon = sample_location!(location_pool)

        state = PatientState(
            Location("FIELD$attempts", latlon, -1, FIELD),
            sample_onset_time(),
            UNKNOWN,
            sample_stroke_type(mdp),
        )

        # Try all four policies. If any one fails, record dropout (with the
        # specific failing policy + reason) and move on. Paired comparison is
        # preserved by requiring all four to succeed for the patient to be
        # included in the main results CSV; dropouts.csv exposes the
        # selection step for downstream sensitivity analysis.
        policy_outputs = Vector{Tuple{String, Float64, Float64}}(undef, 4)
        failure        = nothing

        for (i, (name, policy_fn)) in enumerate(zip(POLICY_NAMES, POLICIES))
            (result, reason) = evaluate_policy(mdp, state, policy_fn)
            if result === nothing
                failure = (name, reason)
                break
            end
            policy_outputs[i] = result
        end

        if failure !== nothing
            push!(dropouts, Dropout(
                attempts, latlon[1], latlon[2],
                state.t_onset, string(state.stroke_type),
                failure[1], failure[2],
            ))
            continue
        end

        push!(results, SimulationResult(
            state,
            ntuple(i -> policy_outputs[i][1], 4),
            ntuple(i -> policy_outputs[i][2], 4),
            ntuple(i -> policy_outputs[i][3], 4),
        ))
        next!(progress)
    end
    finish!(progress)

    return results, dropouts, attempts
end

# ============================================================================
# Output formatting
# ============================================================================
function results_to_dataframe(results)
    return DataFrame(
        start_lat               = [r.start_state.loc.latlon[1]   for r in results],
        start_lon               = [r.start_state.loc.latlon[2]   for r in results],
        t_onset                 = [r.start_state.t_onset         for r in results],
        stroke_type             = [string(r.start_state.stroke_type) for r in results],
        optimal_action          = [r.actions[1]                  for r in results],
        optimal_action_reward   = [r.rewards[1]                  for r in results],
        travel_time_optimal     = [r.travel_times[1]             for r in results],
        nearest_hospital_action = [r.actions[2]                  for r in results],
        nearest_hospital_reward = [r.rewards[2]                  for r in results],
        travel_time_nh          = [r.travel_times[2]             for r in results],
        heuristic_1_action      = [r.actions[3]                  for r in results],
        heuristic_1_reward      = [r.rewards[3]                  for r in results],
        travel_time_h1          = [r.travel_times[3]             for r in results],
        heuristic_2_action      = [r.actions[4]                  for r in results],
        heuristic_2_reward      = [r.rewards[4]                  for r in results],
        travel_time_h2          = [r.travel_times[4]             for r in results],
    )
end

function dropouts_to_dataframe(dropouts)
    return DataFrame(
        sim_attempt   = [d.sim_attempt   for d in dropouts],
        start_lat     = [d.start_lat     for d in dropouts],
        start_lon     = [d.start_lon     for d in dropouts],
        t_onset       = [d.t_onset       for d in dropouts],
        stroke_type   = [d.stroke_type   for d in dropouts],
        failed_policy = [d.failed_policy for d in dropouts],
        reason        = [d.reason        for d in dropouts],
    )
end

# ============================================================================
# Main
# ============================================================================
println("Loading location pool from $POINTS_CSV ...")
# Per-replicate shuffle: Random.seed!(SEED) above sets the RNG state, so this
# shuffle is deterministic given the seed but distinct across replicates.
# The simulation loop pops from the front, giving each replicate 1,000 unique
# patients drawn (effectively) without replacement from the canonical pool.
location_pool = shuffle([(row.Latitude, row.Longitude) for row in CSV.File(POINTS_CSV)])
println("  loaded $(length(location_pool)) candidate locations")
if N_SIMULATIONS > length(location_pool)
    error("N_SIMULATIONS=$N_SIMULATIONS exceeds pool size $(length(location_pool)); " *
          "regenerate CA_points.csv with more samples or lower N_SIMULATIONS.")
end

mdp = StrokeMDP()
println("Running $N_SIMULATIONS CA simulations (seed=$SEED, depth=$MDP_DEPTH)")

results, dropouts, attempts = run_simulation(mdp, location_pool)

mkpath(dirname(RESULTS_CSV))
CSV.write(RESULTS_CSV,   results_to_dataframe(results))
CSV.write(DROPOUTS_CSV,  dropouts_to_dataframe(dropouts))

# --- Summary report -------------------------------------------------------
n_dropped = length(dropouts)
drop_pct  = round(100 * n_dropped / max(attempts, 1), digits = 1)

println()
println("┌─ Simulation summary ─────────────────────────────────────")
println("│  Successful patients:  $(length(results))")
println("│  Dropped patients:     $n_dropped  ($drop_pct% of $attempts attempts)")
if n_dropped > 0
    drop_by_policy = countmap([d.failed_policy for d in dropouts])
    println("│  Dropouts by failing policy:")
    for (policy, count) in sort(collect(drop_by_policy), by = x -> -x[2])
        println("│    $policy: $count")
    end
    drop_by_reason = countmap([d.reason for d in dropouts])
    println("│  Dropouts by reason:")
    for (reason, count) in sort(collect(drop_by_reason), by = x -> -x[2])
        println("│    $reason: $count")
    end
end
println("└──────────────────────────────────────────────────────────")
println("Wrote $RESULTS_CSV")
println("Wrote $DROPOUTS_CSV")
