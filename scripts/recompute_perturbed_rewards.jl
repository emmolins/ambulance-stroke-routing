# =============================================================================
# scripts/recompute_perturbed_rewards.jl
#
# Post-hoc travel-time SENSITIVITY analysis.
#
# Why post-hoc?
#   The dispatcher at t = 0 sees only the ORS travel-time estimate. The actual
#   ambulance travel time is realised AFTER dispatch. The deterministic-travel-
#   time critique is therefore best addressed by:
#     (1) holding the planner's decision fixed (it cannot react to noise it
#         doesn't see), and
#     (2) perturbing the realised travel time and re-evaluating the patient
#         outcome under the perturbed time.
#   This mirrors `recompute_realized_rewards.jl`, which is the same post-hoc
#   logic but with no perturbation (only stroke-type realisation).
#
# What the script does
#   For each per-replicate baseline CSV, and for each (patient, policy) row:
#     1. Read the recorded patient-pickup travel time t.
#     2. Apply systematic bias:  t' = t × TRAVEL_TIME_BIAS_MULT
#     3. Apply log-normal noise: t'' = t' × exp(σ·Z − σ²/2),  Z ~ N(0, 1)
#     4. Recompute the reward via reward(mdp, s, a, sp) using:
#          - the patient's TRUE stroke type (KNOWN branch — realized outcome)
#          - the perturbed travel time t''
#     5. Write the perturbed-reward CSV with an auto-generated tag.
#
# Scope of perturbation
#   Only the patient → first-hospital leg is perturbed (the dispatcher-
#   uncertainty leg). PSC → CSC transfer times remain deterministic ORS values
#   — those are well-known to the medical system and held constant across all
#   sensitivity settings.
#
# Reproducibility
#   The RNG is seeded as a function of (replicate index, σ, bias) so reruns
#   produce identical perturbations.
#
# Inputs (existing)
#   simulation_results/CA_simulation_results_rep<N>.csv       (10 replicates)
#
# Outputs (new)
#   simulation_results/CA_simulation_results_rep<N>_noise<P>.csv
#   simulation_results/CA_simulation_results_rep<N>_bias<P>.csv
#   simulation_results/CA_simulation_results_rep<N>_noise<P>_bias<P>.csv
#   ...depending on which knobs are set.
#
# Run
#   # Variance sensitivity (multiplicative log-normal noise):
#   TRAVEL_TIME_NOISE_SD=0.10 julia --project=. scripts/recompute_perturbed_rewards.jl
#   TRAVEL_TIME_NOISE_SD=0.20 julia --project=. scripts/recompute_perturbed_rewards.jl
#
#   # Systematic-bias sensitivity (ORS-vs-real calibration error):
#   TRAVEL_TIME_BIAS_MULT=0.90 julia --project=. scripts/recompute_perturbed_rewards.jl
#   TRAVEL_TIME_BIAS_MULT=0.80 julia --project=. scripts/recompute_perturbed_rewards.jl
#   TRAVEL_TIME_BIAS_MULT=0.70 julia --project=. scripts/recompute_perturbed_rewards.jl
#   TRAVEL_TIME_BIAS_MULT=1.10 julia --project=. scripts/recompute_perturbed_rewards.jl
#   TRAVEL_TIME_BIAS_MULT=1.20 julia --project=. scripts/recompute_perturbed_rewards.jl
#   TRAVEL_TIME_BIAS_MULT=1.30 julia --project=. scripts/recompute_perturbed_rewards.jl
#
#   # Then pool + summarise:
#   julia --project=. scripts/aggregate_replicates.jl
#   julia --project=. scripts/sensitivity_summary.jl
# =============================================================================

using CSV
using DataFrames
using Random
using Statistics

include("../src/CA_STPMDP_ORS.jl")

const DIR = "simulation_results"

# ----- Sensitivity knobs (read from env, default = no perturbation) ----------
const NOISE_SD  = haskey(ENV, "TRAVEL_TIME_NOISE_SD") ?
                  parse(Float64, ENV["TRAVEL_TIME_NOISE_SD"]) : 0.0
const BIAS_MULT = haskey(ENV, "TRAVEL_TIME_BIAS_MULT") ?
                  parse(Float64, ENV["TRAVEL_TIME_BIAS_MULT"]) : 1.0

if NOISE_SD == 0 && BIAS_MULT == 1.0
    error("Refusing to run with no perturbation: output name would equal the input and overwrite the baseline replicate files. Set TRAVEL_TIME_NOISE_SD and/or TRAVEL_TIME_BIAS_MULT.")
    @warn "Both TRAVEL_TIME_NOISE_SD and TRAVEL_TIME_BIAS_MULT are at their " *
          "no-op defaults; this run will reproduce the realized-rewards baseline " *
          "(equivalent to recompute_realized_rewards.jl)."
end

const NOISE_TAG = NOISE_SD == 0   ? "" : "_noise$(Int(round(100 * NOISE_SD)))"
const BIAS_TAG  = BIAS_MULT == 1  ? "" : "_bias$(Int(round(100 * BIAS_MULT)))"
const TAG       = NOISE_TAG * BIAS_TAG

# (action_col, reward_col, travel_time_col) — schema of the baseline CSVs.
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
Multiplicative log-normal noise factor with unit mean.
Returns 1.0 when σ == 0 so the call site stays branch-free.
"""
@inline function noise_factor(σ::Float64)
    σ == 0 && return 1.0
    return exp(σ * randn() - σ^2 / 2)
end

"""
Recompute realized reward under perturbed travel time.

Same reconstruction as `realized_reward` in recompute_realized_rewards.jl
(KNOWN branch, true stroke type), except the patient → first-hospital travel
time is the perturbed value t_perturbed instead of the recorded one.
"""
function perturbed_reward(mdp, row_idx, lat, lon, t_onset, realized_type,
                          action_str, t_perturbed)
    start_loc = Location("FIELD$row_idx", (lat, lon), -1, FIELD)
    s = PatientState(start_loc, t_onset, KNOWN, realized_type)

    a = string_to_enum(action_str)
    dest_name = replace(action_str, "ROUTE_" => "")
    dest_idx = findfirst(loc -> loc.name == dest_name, mdp.locations)
    dest_idx === nothing && return NaN
    dest_loc = mdp.locations[dest_idx]

    # Reconstruct sp using the PERTURBED first-leg time. Must mirror transition():
    # from the field, arrival time = onset + travel (in-hospital intervals are
    # applied inside reward()).
    sp_t_onset = t_onset + t_perturbed
    sp = PatientState(dest_loc, sp_t_onset, KNOWN, realized_type)

    return reward(mdp, s, a, sp)
end

"""
Seed for a (replicate, σ, bias) triple. Same triple → identical perturbations.
"""
function rng_seed(rep_id::Int, σ::Float64, bias::Float64)
    # Mix the three knobs into a deterministic integer seed. Multiplying by
    # large primes keeps the components separable; integer-cast is safe
    # because we only ever use a finite set of (σ, bias) values.
    return 1234 +
           10_000 * rep_id +
           1_000_000 * round(Int, 1000 * σ) +
           1_000_000_000 * round(Int, 1000 * bias)
end

function process_file(in_path::String, out_path::String, mdp::StrokeMDP,
                       rep_id::Int)
    println("Reading $in_path ...")
    df = CSV.read(in_path, DataFrame)
    n = nrow(df)
    println("  $n rows  |  seeding RNG = $(rng_seed(rep_id, NOISE_SD, BIAS_MULT))")

    Random.seed!(rng_seed(rep_id, NOISE_SD, BIAS_MULT))

    # One noise draw per (patient, destination). Policies that send the same
    # patient to the same hospital must see the same perturbed travel time;
    # otherwise their paired difference is pure noise instead of zero and the
    # paired SE is inflated. Keyed by action string (ROUTE_<hospital>).
    noise_by_dest = [Dict{String, Float64}() for _ in 1:n]
    draw_noise(i, action_str) = get!(noise_by_dest[i], String(action_str)) do
        noise_factor(NOISE_SD)
    end

    for (action_col, reward_col, travel_col) in POLICY_SPECS
        new_rewards = Vector{Float64}(undef, n)
        for i in 1:n
            realized_type = STROKE_TYPE_MAP[df[i, :stroke_type]]
            t_recorded    = df[i, travel_col]
            t_perturbed   = t_recorded * BIAS_MULT * draw_noise(i, df[i, action_col])
            new_rewards[i] = perturbed_reward(
                mdp, i,
                df[i, :start_lat], df[i, :start_lon],
                df[i, :t_onset],
                realized_type,
                df[i, action_col],
                t_perturbed,
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
println("="^78)
println("POST-HOC TRAVEL-TIME PERTURBATION")
println("="^78)
println("  σ (log-normal noise) : $NOISE_SD")
println("  bias multiplier      : $BIAS_MULT")
println("  output tag           : $(isempty(TAG) ? "(none — baseline)" : TAG)")
println()

println("Building MDP (loading hospitals)...")
mdp = StrokeMDP()
println("  $(length(mdp.locations)) locations loaded")
println()

rep_pattern = r"^CA_simulation_results_rep(\d+)\.csv$"
rep_files = filter(f -> occursin(rep_pattern, f), readdir(DIR))
sort!(rep_files; by = f -> parse(Int, match(rep_pattern, f).captures[1]))

isempty(rep_files) &&
    error("No CA_simulation_results_rep*.csv files found in $DIR. " *
          "Run the baseline simulation first.")

println("Processing $(length(rep_files)) per-replicate baseline file(s) ...")
for f in rep_files
    rep_id = parse(Int, match(rep_pattern, f).captures[1])
    in_path  = joinpath(DIR, f)
    out_name = replace(f, ".csv" => "$(TAG).csv")
    out_path = joinpath(DIR, out_name)
    process_file(in_path, out_path, mdp, rep_id)
end

println("="^78)
println("Done.")
println()
println("Next:")
println("  julia --project=. scripts/aggregate_replicates.jl    # pool the new files")
println("  julia --project=. scripts/sensitivity_summary.jl     # robustness table")
