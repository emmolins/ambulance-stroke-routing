# =============================================================================
# scripts/sensitivity_summary.jl
#
# Travel-time sensitivity summary table.
#
# Reads pooled simulation outputs across BOTH sensitivity axes:
#   1. Variance (TRAVEL_TIME_NOISE_SD): log-normal multiplicative noise on
#      every ORS call.
#   2. Systematic bias (TRAVEL_TIME_BIAS_MULT): deterministic multiplicative
#      shift on every ORS call (calibration error between ORS and real EMS times).
#
# Produces a manuscript-ready comparison of policy contrasts at every
# sensitivity setting. The point of the table is to show reviewers that the
# headline policy differences are robust to plausible travel-time mis-modelling.
#
# Input files (any subset present — script reports what it finds):
#   simulation_results/CA_simulation_results_realized.csv             (baseline; realized rewards,
#                                                                      same definition as the perturbed rows)
#   simulation_results/CA_simulation_results_noise10.csv              (10% noise)
#   simulation_results/CA_simulation_results_noise20.csv              (20% noise)
#   simulation_results/CA_simulation_results_bias90.csv               (×0.90 bias)
#   simulation_results/CA_simulation_results_bias110.csv              (×1.10 bias)
#   simulation_results/CA_simulation_results_bias70.csv               (×0.70 bias)
#   simulation_results/CA_simulation_results_bias130.csv              (×1.30 bias)
#   simulation_results/CA_simulation_results_noise10_bias110.csv      (combined)
#   ...
#
# Output:
#   simulation_results/CA_sensitivity_summary.csv  — one row per (setting, comparison)
#
# Run:
#     julia --project=. scripts/sensitivity_summary.jl
# =============================================================================

using CSV
using DataFrames
using Statistics
using Distributions
using Printf

const DIR = "simulation_results"

# Policy contrasts computed at every sensitivity setting.
const COMPARISONS = [
    ("MDP optimal − Nearest",     :optimal_action_reward, :nearest_hospital_reward),
    ("MDP optimal − Heuristic 1", :optimal_action_reward, :heuristic_1_reward),
    ("MDP optimal − Heuristic 2", :optimal_action_reward, :heuristic_2_reward),
    ("Heuristic 1 − Nearest",     :heuristic_1_reward,    :nearest_hospital_reward),
    ("Heuristic 2 − Nearest",     :heuristic_2_reward,    :nearest_hospital_reward),
]

# Recognises pooled files with any combination of noise/bias tags.
const POOLED_PATTERN = r"^CA_simulation_results(_noise(\d+))?(_bias(\d+))?\.csv$"

"""
Discover pooled sensitivity files. Returns a vector of
NamedTuples (noise_pct, bias_pct, path), sorted for stable presentation
(baseline first, then noise-only by level, then bias-only by level,
then combined).
"""
function find_sensitivity_files(dir)
    out = NamedTuple[]
    for f in readdir(dir)
        m = match(POOLED_PATTERN, f)
        m === nothing && continue
        noise_pct = m.captures[2] === nothing ? 0   : parse(Int, m.captures[2])
        bias_pct  = m.captures[4] === nothing ? 100 : parse(Int, m.captures[4])
        path = joinpath(dir, f)
        if noise_pct == 0 && bias_pct == 100
            # The perturbed files hold REALIZED rewards (recompute_perturbed_rewards.jl
            # evaluates the KNOWN-type branch). The baseline row must use the same
            # reward definition, i.e. CA_simulation_results_realized.csv, not the
            # marginalised rewards in CA_simulation_results.csv.
            realized = joinpath(dir, "CA_simulation_results_realized.csv")
            if isfile(realized)
                path = realized
            else
                @warn "Baseline skipped: $(basename(realized)) not found. Run scripts/recompute_realized_rewards.jl first; the marginalised baseline is not comparable with the perturbed (realized) rows."
                continue
            end
        end
        push!(out, (noise_pct = noise_pct, bias_pct = bias_pct, path = path))
    end
    # Sort: baseline first; then noise-only; then bias-only; then combined.
    sort!(out; by = r -> (r.noise_pct > 0 && r.bias_pct != 100,  # combined last
                          r.bias_pct != 100,                      # bias before combined
                          r.noise_pct > 0,                        # noise before bias
                          r.noise_pct,
                          r.bias_pct))
    return out
end

function paired_stats(x::AbstractVector, y::AbstractVector)
    diffs = collect(Float64.(x .- y))
    n = length(diffs)
    mean_d = mean(diffs)
    sd_d = std(diffs)
    se = sd_d > 0 ? sd_d / sqrt(n) : 0.0
    t_crit = quantile(TDist(n - 1), 0.975)
    return (
        n = n,
        mean = mean_d,
        se = se,
        ci_lo = mean_d - t_crit * se,
        ci_hi = mean_d + t_crit * se,
    )
end

function setting_label(noise_pct, bias_pct)
    if noise_pct == 0 && bias_pct == 100
        return "baseline"
    end
    parts = String[]
    noise_pct > 0   && push!(parts, "noise=$(noise_pct)%")
    bias_pct != 100 && push!(parts, "bias=×$(bias_pct/100)")
    return join(parts, ", ")
end

function build_table(files)
    rows = NamedTuple[]
    for f in files
        df = CSV.read(f.path, DataFrame)
        for (label, col_a, col_b) in COMPARISONS
            (hasproperty(df, col_a) && hasproperty(df, col_b)) || continue
            s = paired_stats(df[!, col_a], df[!, col_b])
            push!(rows, (
                noise_pct  = f.noise_pct,
                bias_pct   = f.bias_pct,
                setting    = setting_label(f.noise_pct, f.bias_pct),
                comparison = label,
                n = s.n,
                mean_diff = s.mean,
                se = s.se,
                ci95_lo = s.ci_lo,
                ci95_hi = s.ci_hi,
            ))
        end
    end
    return DataFrame(rows)
end

# ============================================================================
# Main
# ============================================================================
files = find_sensitivity_files(DIR)
isempty(files) && error("No CA_simulation_results*.csv files found in $DIR.")

println("="^100)
println("TRAVEL-TIME SENSITIVITY SUMMARY")
println("="^100)
println("Found $(length(files)) sensitivity setting(s):")
for f in files
    n = countlines(f.path) - 1
    println("  $(rpad(setting_label(f.noise_pct, f.bias_pct), 30))  $(basename(f.path))   ($n patients)")
end
println()

tbl = build_table(files)

# Pretty-print: one column block per setting for each comparison
println("="^120)
@printf("%-30s %-22s %5s %12s %12s %22s\n",
    "Comparison", "Setting", "n", "mean Δ", "SE", "95% CI")
println("-"^120)
for label in unique(tbl.comparison)
    for row in eachrow(tbl[tbl.comparison .== label, :])
        @printf("%-30s %-22s %5d %12.5f %12.5f   [%.5f, %.5f]\n",
            row.comparison, row.setting, row.n,
            row.mean_diff, row.se, row.ci95_lo, row.ci95_hi)
    end
    println()
end

out = joinpath(DIR, "CA_sensitivity_summary.csv")
CSV.write(out, tbl)
println("✓ Wrote $out")
println()
println("Interpretation:")
println("  Variance sensitivity (noise rows): if the baseline mean Δ falls inside")
println("  the noise=10%/20% CIs, the conclusion is robust to plausible travel-")
println("  time variance.")
println("  Bias sensitivity (bias rows): if the policy ranking and significance")
println("  hold across bias ×0.7–×1.3, the conclusion is robust to ±30% ORS")
println("  calibration error vs. real ambulance times.")
