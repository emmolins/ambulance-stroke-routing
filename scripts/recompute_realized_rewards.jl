# =============================================================================
# recompute_realized_rewards.jl
#
# Post-process the existing per-replicate simulation CSVs to add a second
# "realized" reward column per policy. The realized reward conditions on the
# patient's actual sampled stroke type (the KNOWN branch of `reward(...)`),
# rather than the population-marginalized expectation (the UNKNOWN branch).
#
# IMPORTANT — what this does NOT change
#   The MDP planner is unchanged. Action choices are unchanged. `best_action`
#   still uses marginalized expected reward at decision time, which is the
#   correct decision-theoretic approach under stroke-type uncertainty.
#
# What this DOES change
#   The reported per-patient outcome. Realized outcomes are the standard
#   clinical-paper metric: "what actually happened to this patient given their
#   true diagnosis," rather than "what the planner expected to happen averaged
#   over possible diagnoses."
#
# Why both versions are useful
#   - Marginalized: matches the MDP objective (decision-time perspective).
#                   Headline mean policy differences are correct under this.
#   - Realized:     standard clinical reporting (outcome perspective). Same
#                   mean (by linearity) but variance structure differs;
#                   subgroup-by-stroke-type analysis becomes meaningful
#                   (HEM/MIMIC subgroups correctly show 0 policy effect,
#                   LVO/nLVO subgroups concentrate the benefit).
#
# Inputs (existing)
#   simulation_results/CA_simulation_results_rep<N>.csv       (10 replicates)
#   simulation_results/CA_simulation_results.csv              (aggregated, 10000 rows)
#
# Outputs (new)
#   simulation_results/CA_simulation_results_realized_rep<N>.csv
#   simulation_results/CA_simulation_results_realized.csv
#
# Run with:
#   julia --project=. recompute_realized_rewards.jl
#
# Requirements
#   ORS must be running with the CA road graph. The reward function makes
#   PSC->CSC transfer-time calls internally, but these are cached after the
#   first call per PSC, so only ~10-20 ORS calls happen in total.
# =============================================================================

using CSV
using DataFrames

include("../src/CA_STPMDP_ORS.jl")

const DIR = "simulation_results"
# Optional suffix for sensitivity variants, e.g. REALIZED_TAG=_dido60 with DIDO_MIN=60.
const REALIZED_TAG = get(ENV, "REALIZED_TAG", "")

# (action_col, reward_col, travel_time_col) tuples in the order the simulation
# CSVs use. The `reward_col` will be REPLACED with realized values in the
# output file; all other columns are passed through unchanged.
const POLICY_SPECS = [
    (:optimal_action,          :optimal_action_reward,   :travel_time_optimal),
    (:nearest_hospital_action, :nearest_hospital_reward, :travel_time_nh),
    (:heuristic_1_action,      :heuristic_1_reward,      :travel_time_h1),
    (:heuristic_2_action,      :heuristic_2_reward,      :travel_time_h2),
]

const STROKE_TYPE_MAP = Dict(
    "LVO"         => LVO,
    "NLVO"        => NLVO,
    "HEMORRHAGIC" => HEMORRHAGIC,
    "MIMIC"       => MIMIC,
)

"""
Recompute the realized reward for a single (patient, policy) pair.

The planner saw a marginalized expectation when it chose `action_str`. Here we
ask: "given the patient's ACTUAL stroke type, and given the planner's chosen
action, what is the realized probability of good outcome?"

We reconstruct sp (the next state) using the travel time already stored in the
CSV — no fresh ORS call is needed for the transition itself. (Internal
PSC->CSC lookups inside `reward()` may still hit ORS, but those are cached.)
"""
function realized_reward(mdp, row_idx, lat, lon, t_onset, realized_type,
                          action_str, travel_time)
    # Start state with KNOWN diagnosis = the realized type.
    start_loc = Location("FIELD$row_idx", (lat, lon), -1, FIELD)
    s = PatientState(start_loc, t_onset, KNOWN, realized_type)

    # Map action string to enum and destination Location.
    a = string_to_enum(action_str)
    dest_name = replace(action_str, "ROUTE_" => "")
    dest_idx = findfirst(loc -> loc.name == dest_name, mdp.locations)
    dest_idx === nothing && return NaN
    dest_loc = mdp.locations[dest_idx]

    # Reconstruct sp using the stored travel time (deterministic case; with
    # noise > 0 the stored value is the realized noisy time the simulation
    # used, which is still the right thing to feed back into reward()).
    # Mirrors transition(): from the field, arrival time = onset + travel;
    # in-hospital intervals (DTN / DTP / DIDO) are applied inside reward().
    sp_t_onset = t_onset + travel_time
    sp = PatientState(dest_loc, sp_t_onset, KNOWN, realized_type)

    return action_value(mdp, s, a, sp)
end

function process_file(in_path::String, out_path::String, mdp::StrokeMDP)
    println("Reading $in_path ...")
    df = CSV.read(in_path, DataFrame)
    n = nrow(df)
    println("  $n rows")

    for (action_col, reward_col, travel_col) in POLICY_SPECS
        new_rewards = Vector{Float64}(undef, n)
        for i in 1:n
            realized_type = STROKE_TYPE_MAP[df[i, :stroke_type]]
            new_rewards[i] = realized_reward(
                mdp,
                i,
                df[i, :start_lat],
                df[i, :start_lon],
                df[i, :t_onset],
                realized_type,
                df[i, action_col],
                df[i, travel_col],
            )
        end
        df[!, reward_col] = new_rewards
        println("  ✓ recomputed $reward_col  (mean = $(round(mean(new_rewards), digits=4)))")
    end

    CSV.write(out_path, df)
    println("  → wrote $out_path")
    println()
end

# ============================================================================
# Main
# ============================================================================

# Need `mean` from Statistics for the diagnostic prints above.
using Statistics

println("Building MDP (loading hospitals)...")
mdp = StrokeMDP()
println("  $(length(mdp.locations)) locations loaded")
println()

# Per-replicate files
rep_pattern = r"^CA_simulation_results_rep(\d+)\.csv$"
rep_files = filter(f -> occursin(rep_pattern, f), readdir(DIR))
sort!(rep_files; by = f -> parse(Int, match(rep_pattern, f).captures[1]))

if !isempty(rep_files)
    println("Processing $(length(rep_files)) per-replicate file(s) ...")
    for f in rep_files
        in_path  = joinpath(DIR, f)
        out_path = joinpath(DIR, replace(f, "_rep" => "_realized$(REALIZED_TAG)_rep"))
        process_file(in_path, out_path, mdp)
    end
end

# Aggregated file
agg_in  = joinpath(DIR, "CA_simulation_results.csv")
agg_out = joinpath(DIR, "CA_simulation_results_realized$(REALIZED_TAG).csv")
if isfile(agg_in)
    println("Processing aggregated file ...")
    process_file(agg_in, agg_out, mdp)
end

println("=" ^ 78)
println("Done.")
println()
println("To run stats on the REALIZED version (alongside the existing marginalized):")
println()
println("    INPUT_PREFIX=CA_simulation_results_realized julia --project=. CA_simulations_stats.jl")
println()
println("The stats script (modified to accept INPUT_PREFIX) will read the new")
println("realized files and tag its outputs accordingly, so the marginalized")
println("results are preserved.")
