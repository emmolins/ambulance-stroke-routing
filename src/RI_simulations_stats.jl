# =============================================================================
# RI_simulations_stats.jl
# Statistical analysis of Rhode Island stroke-triage simulation results,
# parallel to CA_simulations_stats.jl.
#
# Reads:   simulation_results/RI_simulation_results[_realized][_rep<N>].csv
# Writes:  simulation_results/RI_paired_comparisons[_realized].csv
#          simulation_results/RI_paired_comparisons_by_stroke_type[_realized].csv
#          simulation_results/RI_paired_comparisons_by_region[_realized].csv
#          simulation_results/RI_paired_comparisons_multirep[_realized].csv
#
# Run on marginalized (default):
#     julia --project=. src/RI_simulations_stats.jl
# Run on realized:
#     INPUT_PREFIX=RI_simulation_results_realized julia --project=. src/RI_simulations_stats.jl
# =============================================================================

using CSV, Statistics, Distributions, DataFrames, StatsBase, Printf

# -----------------------------------------------------------------------------
# Input/output prefix configuration via env var
# -----------------------------------------------------------------------------
const INPUT_PREFIX = get(ENV, "INPUT_PREFIX", "RI_simulation_results")
const OUTPUT_TAG   = INPUT_PREFIX == "RI_simulation_results" ? "" : "_realized"
println("Input prefix : $INPUT_PREFIX")
println("Output tag   : '$OUTPUT_TAG'")
println()

# =============================================================================
# DATA LOADING
# =============================================================================

function load_data()
    path = "simulation_results/$(INPUT_PREFIX).csv"
    df = CSV.read(path, DataFrame)
    println("✓ Loaded $(basename(path)) with $(nrow(df)) observations")
    return df
end

# =============================================================================
# PRIMARY ANALYSIS — paired t-test confidence intervals for each policy pair.
# =============================================================================

function paired_comparison(label::String, x::AbstractVector, y::AbstractVector)
    diffs = collect(Float64.(x .- y))
    n = length(diffs)
    mean_d = mean(diffs)
    sd_d = std(diffs)
    se = sd_d > 0 ? sd_d / sqrt(n) : 0.0
    t_stat = sd_d > 0 ? mean_d / se : 0.0
    p_val = sd_d > 0 ? 2 * (1 - cdf(TDist(n - 1), abs(t_stat))) : 1.0
    t_crit = quantile(TDist(n - 1), 0.975)
    cohens_d = sd_d > 0 ? mean_d / sd_d : 0.0
    return (
        comparison = label,
        n          = n,
        mean_diff  = mean_d,
        sd_diff    = sd_d,
        se         = se,
        t          = t_stat,
        df         = n - 1,
        p          = p_val,
        ci95_lo    = mean_d - t_crit * se,
        ci95_hi    = mean_d + t_crit * se,
        cohens_d   = cohens_d,
    )
end

function pairwise_paired_tests(df)
    println("="^120)
    println("PRIMARY ANALYSIS — paired mean-reward comparisons (paired t-test, 95% CI)")
    println("="^120)

    comparisons = NamedTuple[]
    push!(comparisons,
        paired_comparison("MDP optimal − Nearest",
                          df.optimal_action_reward, df.nearest_hospital_reward))
    if hasproperty(df, :heuristic_1_reward)
        push!(comparisons,
            paired_comparison("MDP optimal − Heuristic 1",
                              df.optimal_action_reward, df.heuristic_1_reward))
    end
    if hasproperty(df, :heuristic_2_reward)
        push!(comparisons,
            paired_comparison("MDP optimal − Heuristic 2",
                              df.optimal_action_reward, df.heuristic_2_reward))
    end
    if hasproperty(df, :heuristic_1_reward)
        push!(comparisons,
            paired_comparison("Heuristic 1 − Nearest",
                              df.heuristic_1_reward, df.nearest_hospital_reward))
    end
    if hasproperty(df, :heuristic_2_reward)
        push!(comparisons,
            paired_comparison("Heuristic 2 − Nearest",
                              df.heuristic_2_reward, df.nearest_hospital_reward))
    end

    n_tests = length(comparisons)
    summary = DataFrame(comparisons)
    summary.p_bonferroni = min.(1.0, summary.p .* n_tests)

    @printf("%-28s %6s %12s %12s %10s %14s %14s %10s %22s\n",
        "Comparison", "n", "mean Δ", "SE", "t",
        "p (raw)", "p (Bonf.)", "Cohen's d", "95% CI")
    println("-"^140)
    for row in eachrow(summary)
        @printf("%-28s %6d %12.5f %12.5f %10.3f %14.3g %14.3g %10.3f   [%.5f, %.5f]\n",
            row.comparison, row.n, row.mean_diff, row.se, row.t,
            row.p, row.p_bonferroni, row.cohens_d, row.ci95_lo, row.ci95_hi)
    end
    println()

    out_csv = "simulation_results/RI_paired_comparisons$(OUTPUT_TAG).csv"
    mkpath(dirname(out_csv))
    CSV.write(out_csv, summary)
    println("✓ Saved comparison table to $out_csv")
    println()

    return summary
end

# =============================================================================
# SUBGROUP ANALYSES — by stroke type and by region (rurality)
# =============================================================================

function subgroup_paired_tests(df, subgroup_col::Symbol; label_for_print="subgroup",
                                min_n::Int = 10, save_path::String = "")
    levels = sort(unique(skipmissing(df[!, subgroup_col])))
    all_rows = DataFrame()

    println("="^120)
    println("SUBGROUP ANALYSIS — $label_for_print  (column: $subgroup_col)")
    println("="^120)

    for lvl in levels
        sub = df[df[!, subgroup_col] .== lvl, :]
        n = nrow(sub)
        if n < min_n
            println("\nSkipping $subgroup_col = $lvl: only n = $n (< $min_n)")
            continue
        end
        println("\n────── $subgroup_col = $lvl  (n = $n) ──────")
        summary = pairwise_paired_tests(sub)
        summary[!, :subgroup_column] .= string(subgroup_col)
        summary[!, :subgroup_level]  .= string(lvl)
        all_rows = vcat(all_rows, summary; cols = :union)
    end

    if !isempty(save_path) && nrow(all_rows) > 0
        mkpath(dirname(save_path))
        CSV.write(save_path, all_rows)
        println("✓ Saved combined subgroup table to $save_path")
    end
    return all_rows
end

# Discrete rurality column derived from travel_time_nh.
function add_region_column!(df; col=:region, source=:travel_time_nh)
    if !hasproperty(df, source)
        @warn "Cannot derive region: $source column missing"
        return df
    end
    df[!, col] = map(df[!, source]) do t
        t < 10  ? "urban"    :
        t < 25  ? "suburban" :
                  "rural"
    end
    return df
end

# =============================================================================
# MULTI-REPLICATE AGGREGATION — IEEE-canonical R-replicate protocol
# =============================================================================

function aggregate_replicates(;
        dir = "simulation_results",
        pattern = Regex("^" * INPUT_PREFIX * "_rep(\\d+)\\.csv\$"),
        save_path = "simulation_results/RI_paired_comparisons_multirep$(OUTPUT_TAG).csv")

    files = filter(f -> occursin(pattern, f), readdir(dir))
    sort!(files; by = f -> parse(Int, match(pattern, f).captures[1]))
    isempty(files) && error("No replicate CSVs matching $pattern found in $dir")

    println()
    println("="^120)
    println("MULTI-REPLICATE PRIMARY ANALYSIS — $(length(files)) replicates × (within-replicate N)")
    println("="^120)
    println("Replicate files: " * join(files, ", "))

    rep_dfs = [CSV.read(joinpath(dir, f), DataFrame) for f in files]
    n_reps = length(rep_dfs)

    comparisons = [
        ("MDP optimal − Nearest",     :optimal_action_reward, :nearest_hospital_reward),
        ("MDP optimal − Heuristic 1", :optimal_action_reward, :heuristic_1_reward),
        ("MDP optimal − Heuristic 2", :optimal_action_reward, :heuristic_2_reward),
        ("Heuristic 1 − Nearest",     :heuristic_1_reward,    :nearest_hospital_reward),
        ("Heuristic 2 − Nearest",     :heuristic_2_reward,    :nearest_hospital_reward),
    ]

    rows = NamedTuple[]
    df_perrep_means = DataFrame(comparison = String[], replicate = Int[],
                                 n = Int[], mean_diff = Float64[])

    t_crit = quantile(TDist(n_reps - 1), 0.975)

    for (label, col_a, col_b) in comparisons
        per_rep = Float64[]
        for (i, rep) in enumerate(rep_dfs)
            if !hasproperty(rep, col_a) || !hasproperty(rep, col_b)
                continue
            end
            diffs = rep[!, col_a] .- rep[!, col_b]
            mean_d = mean(diffs)
            push!(per_rep, mean_d)
            push!(df_perrep_means, (label, i, length(diffs), mean_d))
        end
        isempty(per_rep) && continue
        grand_mean       = mean(per_rep)
        inter_rep_sd     = length(per_rep) > 1 ? std(per_rep) : 0.0
        inter_rep_se     = inter_rep_sd / sqrt(length(per_rep))
        ci_lo            = grand_mean - t_crit * inter_rep_se
        ci_hi            = grand_mean + t_crit * inter_rep_se
        push!(rows, (
            comparison         = label,
            n_replicates       = length(per_rep),
            grand_mean         = grand_mean,
            inter_rep_sd       = inter_rep_sd,
            inter_rep_se       = inter_rep_se,
            ci95_lo            = ci_lo,
            ci95_hi            = ci_hi,
            per_rep_min        = minimum(per_rep),
            per_rep_max        = maximum(per_rep),
        ))
    end

    summary = DataFrame(rows)

    @printf("\n%-30s %4s %12s %12s %12s %22s\n",
        "Comparison", "R", "grand Δ", "between-SD", "between-SE", "95% CI (between)")
    println("-"^120)
    for r in eachrow(summary)
        @printf("%-30s %4d %12.5f %12.5f %12.5f   [%.5f, %.5f]\n",
            r.comparison, r.n_replicates, r.grand_mean,
            r.inter_rep_sd, r.inter_rep_se, r.ci95_lo, r.ci95_hi)
    end
    println()

    mkpath(dirname(save_path))
    CSV.write(save_path, summary)
    println("✓ Wrote multi-replicate summary to $save_path")

    perrep_path = replace(save_path, r"\.csv$" => "_perreplicate.csv")
    CSV.write(perrep_path, df_perrep_means)
    println("✓ Wrote per-replicate means to $perrep_path")
    println()

    return summary, df_perrep_means
end

# =============================================================================
# REWARD AND TRAVEL-TIME SUMMARIES
# =============================================================================

function reward_summary(df)
    println("="^70)
    println("REWARD SUMMARY (mean / median / SD / SE per policy)")
    println("="^70)
    policies = [
        ("Optimal Policy",   :optimal_action_reward),
        ("Nearest Hospital", :nearest_hospital_reward),
        ("Heuristic 1",      :heuristic_1_reward),
        ("Heuristic 2",      :heuristic_2_reward),
    ]
    @printf("%-22s %10s %10s %10s %10s\n", "Policy", "Mean", "Median", "SD", "SE")
    println("-"^70)
    for (label, col) in policies
        hasproperty(df, col) || continue
        v = df[!, col]
        @printf("%-22s %10.4f %10.4f %10.4f %10.4f\n",
            label, mean(v), median(v), std(v), std(v)/sqrt(length(v)))
    end
    println()
end

function travel_time_summary(df)
    println("="^70)
    println("TRAVEL TIME SUMMARY (mean / median / SD / SE per policy, minutes)")
    println("="^70)
    cols = [
        ("Optimal",          :travel_time_optimal),
        ("Nearest Hospital", :travel_time_nh),
        ("Heuristic 1",      :travel_time_h1),
        ("Heuristic 2",      :travel_time_h2),
    ]
    @printf("%-22s %10s %10s %10s %10s\n", "Policy", "Mean", "Median", "SD", "SE")
    println("-"^70)
    for (label, col) in cols
        hasproperty(df, col) || continue
        v = df[!, col]
        @printf("%-22s %10.2f %10.2f %10.2f %10.2f\n",
            label, mean(v), median(v), std(v), std(v)/sqrt(length(v)))
    end
    println()
end

# =============================================================================
# MAIN
# =============================================================================

function run_complete_analysis()
    println("Starting Rhode Island stroke-triage simulation analysis ...")
    println()

    df = load_data()
    println()

    pairwise_paired_tests(df)

    if hasproperty(df, :stroke_type)
        subgroup_paired_tests(df, :stroke_type;
            label_for_print = "by stroke type",
            save_path = "simulation_results/RI_paired_comparisons_by_stroke_type$(OUTPUT_TAG).csv")
    end
    add_region_column!(df)
    if hasproperty(df, :region)
        subgroup_paired_tests(df, :region;
            label_for_print = "by region (rurality, proxied by nearest-hospital travel time)",
            save_path = "simulation_results/RI_paired_comparisons_by_region$(OUTPUT_TAG).csv")
    end

    try
        files = filter(f -> occursin(Regex("^" * INPUT_PREFIX * "_rep\\d+\\.csv\$"), f),
                       readdir("simulation_results"))
        if length(files) >= 2
            aggregate_replicates()
        end
    catch e
        @warn "Multi-replicate aggregation skipped: $e"
    end

    reward_summary(df)
    travel_time_summary(df)

    println("="^70)
    println("ANALYSIS COMPLETE")
    println("="^70)

    return df
end

run_complete_analysis()
