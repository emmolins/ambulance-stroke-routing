# =============================================================================
# scripts/equity_analysis.jl
#
# Equity-stratified analysis of the Bay Area grid patient outcomes.
#
# Inputs
#   sampled_points/CA_grid_patients_with_demographics.csv
#       (from scripts/grid_to_tract_join.py)
#
# Strata
#   1. Income quartiles
#        Q1 = bottom 25% by tract median household income
#        Q2, Q3, Q4 = increasing quartiles
#        Quartile boundaries computed over tracts (each tract counted once),
#        then patients inherit their tract's quartile.
#   2. RUCA urban / rural
#        urban  = RUCA 1–3
#        rural  = RUCA 4–10
#
# Output
#   simulation_results/CA_equity_summary.csv
#       One row per (stratum, comparison). For each cell:
#           mean_diff, se, ci95_lo, ci95_hi, n_patients, n_tracts, p_value
#
# Run
#   julia --project=. scripts/equity_analysis.jl
# =============================================================================

using CSV
using DataFrames
using Statistics
using Distributions
using Printf

const IN_CSV  = "sampled_points/CA_grid_patients_with_demographics.csv"
const OUT_CSV = "simulation_results/CA_equity_summary.csv"

const COMPARISONS = [
    ("MDP optimal − Nearest",     :optimal_action_reward, :nearest_hospital_reward),
    ("MDP optimal − Heuristic 1", :optimal_action_reward, :heuristic_1_reward),
    ("MDP optimal − Heuristic 2", :optimal_action_reward, :heuristic_2_reward),
    ("Heuristic 1 − Nearest",     :heuristic_1_reward,    :nearest_hospital_reward),
    ("Heuristic 2 − Nearest",     :heuristic_2_reward,    :nearest_hospital_reward),
]

# ----- Paired-difference stats (matches scripts/sensitivity_summary.jl) -----
function paired_stats(x::AbstractVector, y::AbstractVector)
    diffs = collect(Float64.(x .- y))
    n = length(diffs)
    n < 2 && return (n = n, mean = NaN, se = NaN, ci_lo = NaN, ci_hi = NaN,
                     p = NaN)
    mean_d = mean(diffs)
    sd_d = std(diffs)
    se = sd_d > 0 ? sd_d / sqrt(n) : 0.0
    if se == 0
        return (n = n, mean = mean_d, se = 0.0, ci_lo = mean_d, ci_hi = mean_d,
                p = 1.0)
    end
    t = mean_d / se
    t_crit = quantile(TDist(n - 1), 0.975)
    # Two-sided p-value via t distribution.
    p = 2 * ccdf(TDist(n - 1), abs(t))
    return (
        n = n,
        mean = mean_d,
        se = se,
        ci_lo = mean_d - t_crit * se,
        ci_hi = mean_d + t_crit * se,
        p = p,
    )
end

# ----- Per-stratum: run all five contrasts ----------------------------------
function stratum_table(df::DataFrame, axis::String, value::String)
    n_tracts = length(unique(df.GEOID))
    out = NamedTuple[]
    for (label, col_a, col_b) in COMPARISONS
        s = paired_stats(df[!, col_a], df[!, col_b])
        push!(out, (
            axis       = axis,
            stratum    = value,
            comparison = label,
            n_patients = s.n,
            n_tracts   = n_tracts,
            mean_diff  = s.mean,
            se         = s.se,
            ci95_lo    = s.ci_lo,
            ci95_hi    = s.ci_hi,
            p_value    = s.p,
        ))
    end
    return out
end

# ----- Income quartile assignment -------------------------------------------
"""
Per-tract income quartile, computed over the unique set of tracts in the
data (so quartile boundaries are tract-weighted, not patient-weighted).
Returns a Dict{GEOID => "Q1"..."Q4"}.
"""
function income_quartile_map(df::DataFrame)
    tract_income = unique(select(df, :GEOID, :median_hh_income))
    tract_income = filter(:median_hh_income => !ismissing, tract_income)
    incomes = collect(Float64.(skipmissing(tract_income.median_hh_income)))
    q = quantile(incomes, [0.25, 0.5, 0.75])
    function bucket(x)
        ismissing(x) && return missing
        xf = Float64(x)
        xf <= q[1] && return "Q1 (lowest)"
        xf <= q[2] && return "Q2"
        xf <= q[3] && return "Q3"
        return "Q4 (highest)"
    end
    return Dict(row.GEOID => bucket(row.median_hh_income)
                for row in eachrow(tract_income)), q
end

# ============================================================================
# Main
# ============================================================================
isfile(IN_CSV) || error("$IN_CSV not found. Run scripts/grid_to_tract_join.py first.")

println("="^78)
println("EQUITY ANALYSIS")
println("="^78)
df = CSV.read(IN_CSV, DataFrame)
println("  Patients : $(nrow(df))")
println("  Tracts   : $(length(unique(df.GEOID)))")
println()

rows = NamedTuple[]

# ----- Axis 1: Income quartile ----------------------------------------------
println("Stratifying by tract median household income (quartiles) ...")
qmap, qbounds = income_quartile_map(df)
@printf("  Q1 cutoff : \$%10.0f\n", qbounds[1])
@printf("  Median    : \$%10.0f\n", qbounds[2])
@printf("  Q3 cutoff : \$%10.0f\n", qbounds[3])

df.income_quartile = [get(qmap, g, missing) for g in df.GEOID]
for q in ("Q1 (lowest)", "Q2", "Q3", "Q4 (highest)")
    sub = df[coalesce.(df.income_quartile .== q, false), :]
    isempty(sub) && continue
    println("  $q : $(nrow(sub)) patients in $(length(unique(sub.GEOID))) tracts")
    append!(rows, stratum_table(sub, "income_quartile", q))
end
println()

# ----- Axis 2: RUCA urban / rural -------------------------------------------
println("Stratifying by RUCA urbanicity ...")
for u in ("urban", "rural")
    sub = df[coalesce.(df.urbanicity .== u, false), :]
    isempty(sub) && continue
    println("  $u : $(nrow(sub)) patients in $(length(unique(sub.GEOID))) tracts")
    append!(rows, stratum_table(sub, "urbanicity", u))
end
println()

# ----- Axis 3 (bonus): urbanicity × income ---------------------------------
# Useful crosstab — does the income gradient differ in urban vs rural settings?
println("Stratifying by urbanicity × income (crosstab) ...")
for u in ("urban", "rural")
    for q in ("Q1 (lowest)", "Q4 (highest)")
        sub = df[coalesce.((df.urbanicity .== u) .& (df.income_quartile .== q), false), :]
        nrow(sub) < 30 && continue
        label = "$u × $q"
        println("  $label : $(nrow(sub)) patients")
        append!(rows, stratum_table(sub, "urban_x_income", label))
    end
end
println()

# ----- Write -----------------------------------------------------------------
tbl = DataFrame(rows)
mkpath(dirname(OUT_CSV))
CSV.write(OUT_CSV, tbl)
println("  ✓ Wrote $OUT_CSV  ($(nrow(tbl)) rows)")
println()

# ----- Pretty-print the table -----------------------------------------------
println("="^120)
@printf("%-20s %-22s %-30s %10s %12s %12s %22s %10s\n",
    "Axis", "Stratum", "Comparison", "n_pat", "mean Δ", "SE", "95% CI", "p")
println("-"^120)
for r in eachrow(tbl)
    @printf("%-20s %-22s %-30s %10d %12.5f %12.5f   [% .5f,% .5f] %10.4g\n",
        r.axis, r.stratum, r.comparison, r.n_patients,
        r.mean_diff, r.se, r.ci95_lo, r.ci95_hi, r.p_value)
end
println()
println("Interpretation:")
println("  Across income quartiles: if MDP advantage is LARGER in Q1 (lowest")
println("  income) than Q4, that's a positive equity finding — optimization")
println("  helps underserved areas more.")
println("  Across urbanicity: if MDP advantage is LARGER in rural than urban,")
println("  the optimization is closing access gaps. Smaller in rural suggests")
println("  the MDP can't compensate for low hospital density.")
