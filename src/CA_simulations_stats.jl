#=
File: CA_simulations_stats.jl
----------------------------------------------------------------------
Performs comprehensive statistical analysis of stroke triage simulation 
results comparing optimal MDP-based ambulance routing to heuristic 
strategies across California. Includes reward comparisons, hypothesis 
testing, travel time evaluation, agreement analysis, and visualizations.
=#

using CSV, Statistics, Distributions, DataFrames, Plots, Measures, StatsBase, Printf
# `Measures` is needed for the `mm` plot-margin units in the histogram block.

# Set default plot styling
default(fontfamily="Times New Roman")

# -----------------------------------------------------------------------------
# Input/output prefix (env var)
#
# Default: "CA_simulation_results" (reads the marginalized files; outputs
#          without a tag).
# To run on realized rewards instead:
#   INPUT_PREFIX=CA_simulation_results_realized julia --project=. CA_simulations_stats.jl
# The script will then read the *_realized*.csv inputs and tag every output
# file with a `_realized` suffix so the two analyses don't clobber each other.
# -----------------------------------------------------------------------------
const INPUT_PREFIX = get(ENV, "INPUT_PREFIX", "CA_simulation_results")
const OUTPUT_TAG   = replace(INPUT_PREFIX, "CA_simulation_results" => "")   # "", "_realized", "_evtonsite", ...
println("Input prefix : $INPUT_PREFIX")
println("Output tag   : '$OUTPUT_TAG'")
println()

# Load and prep data for analysis
function load_and_prepare_data()
    println("="^70)
    println("DATA LOADING AND PREPARATION")
    println("="^70)

    # Load CSV data (path resolved from INPUT_PREFIX env var)
    df = CSV.read("simulation_results/$(INPUT_PREFIX).csv", DataFrame)
    println("✓ Loaded $(INPUT_PREFIX).csv with $(nrow(df)) observations")

    println("✓ Extracting reward data from CSV")
    optimal_action_rewards = df.optimal_action_reward
    nearest_hospital_rewards = df.nearest_hospital_reward
    heuristic_1_rewards = hasproperty(df, :heuristic_1_reward) ? df.heuristic_1_reward : nothing
    heuristic_2_rewards = hasproperty(df, :heuristic_2_reward) ? df.heuristic_2_reward : nothing

    # Check for travel time columns (field names from results_to_dataframe)
    optimal_action_traveltime = hasproperty(df, :travel_time_optimal) ? df.travel_time_optimal : nothing
    nearest_hospital_traveltime = hasproperty(df, :travel_time_nh) ? df.travel_time_nh : nothing
    heuristic_1_traveltime = hasproperty(df, :travel_time_h1) ? df.travel_time_h1 : nothing
    heuristic_2_traveltime = hasproperty(df, :travel_time_h2) ? df.travel_time_h2 : nothing

    return df, optimal_action_rewards, nearest_hospital_rewards, heuristic_1_rewards, heuristic_2_rewards,
    optimal_action_traveltime, nearest_hospital_traveltime, heuristic_1_traveltime, heuristic_2_traveltime
end

# ============================================================================
# PRIMARY ANALYSIS — paired t-test confidence intervals on per-patient
# reward differences, across every policy pair.
#
# For each pair (A, B), we compute:
#   * mean per-patient difference  Δ̄ = mean(A_i − B_i)
#   * 95% CI via the t-distribution: Δ̄ ± t_{n-1,0.975} · SE(Δ)
#   * paired t-test p-value
#   * Bonferroni-adjusted p-value across the family of comparisons
#   * Cohen's d for paired differences (effect size, d = Δ̄ / sd(Δ))
#
# Output is printed and also written to
#   simulation_results/CA_paired_comparisons.csv
# for direct use in the manuscript's results table.
# ============================================================================

# Compute a single paired comparison; returns a NamedTuple.
function paired_comparison(label::String, x::AbstractVector, y::AbstractVector)
    diffs = collect(Float64.(x .- y))
    n = length(diffs)
    mean_d = mean(diffs)
    sd_d = std(diffs)
    se = sd_d > 0 ? sd_d / sqrt(n) : 0.0
    t_stat = sd_d > 0 ? mean_d / se : 0.0
    # Two-sided p-value
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

# Run every pairwise comparison among the four policies, print and save.
function pairwise_paired_tests(df)
    println("="^120)
    println("PRIMARY ANALYSIS — paired mean-reward comparisons (paired t-test, 95% CI)")
    println("="^120)
    println("Each row tests H0: mean per-patient difference (A − B) = 0.")
    println("Effect size = Cohen's d for paired differences. p (Bonf.) = raw p × (number of comparisons).")
    println()

    comparisons = NamedTuple[]

    # MDP-optimal vs each baseline (the headline comparisons)
    push!(comparisons,
        paired_comparison("MDP optimal − Nearest",
                          df.optimal_action_reward,  df.nearest_hospital_reward))
    if hasproperty(df, :heuristic_1_reward)
        push!(comparisons,
            paired_comparison("MDP optimal − Heuristic 1",
                              df.optimal_action_reward,  df.heuristic_1_reward))
    end
    if hasproperty(df, :heuristic_2_reward)
        push!(comparisons,
            paired_comparison("MDP optimal − Heuristic 2",
                              df.optimal_action_reward,  df.heuristic_2_reward))
    end

    # Each heuristic vs status quo (secondary comparisons)
    if hasproperty(df, :heuristic_1_reward)
        push!(comparisons,
            paired_comparison("Heuristic 1 − Nearest",
                              df.heuristic_1_reward,     df.nearest_hospital_reward))
    end
    if hasproperty(df, :heuristic_2_reward)
        push!(comparisons,
            paired_comparison("Heuristic 2 − Nearest",
                              df.heuristic_2_reward,     df.nearest_hospital_reward))
    end

    n_tests = length(comparisons)
    summary = DataFrame(comparisons)
    summary.p_bonferroni = min.(1.0, summary.p .* n_tests)

    # Pretty-print as a table
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

    out_csv = "simulation_results/CA_paired_comparisons$(OUTPUT_TAG).csv"
    mkpath(dirname(out_csv))
    CSV.write(out_csv, summary)
    println("✓ Saved comparison table to $out_csv")
    println()

    return summary
end

# ============================================================================
# SUBGROUP ANALYSES — equity-relevant breakdowns of the primary analysis
#
# Reviewers asked for stratified results so we can ask:
#   * Is the MDP advantage concentrated in particular stroke types?
#   * Is the MDP advantage concentrated in particular geographic regions?
#
# Both are addressed by running `pairwise_paired_tests` within each subgroup
# level and concatenating the output tables. The full subgroup summary is
# also saved to a single CSV for the manuscript's supplementary tables.
# ============================================================================

# Run pairwise_paired_tests separately within each level of `subgroup_col` and
# return a stacked summary DataFrame with the subgroup label added as a column.
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

# Add a discrete rurality column derived from the travel_time_nh column (the
# travel time the patient would have had under the nearest-hospital policy).
# This is a defensible proxy for geographic isolation that requires no external
# data: short travel-to-nearest indicates urban access, long indicates rural.
#
# Cutpoints (in minutes) match common EMS reporting conventions:
#   urban    : nearest hospital within 10 min
#   suburban : 10 – 25 min
#   rural    : >25 min
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

# ============================================================================
# MULTI-REPLICATE AGGREGATION — for the IEEE-style "R independent replicates"
# protocol. Reads every `CA_simulation_results_rep<N>.csv` in the results
# folder, computes the within-replicate mean difference for each policy pair,
# then reports the grand mean across replicates with inter-replicate SE and
# t-distributed 95% CI on (R-1) degrees of freedom.
#
# This is what reviewers expect to see when you assert "results are robust
# across N independent simulations with different random seeds."
# ============================================================================
function aggregate_replicates(;
        dir = "simulation_results",
        pattern = Regex("^" * INPUT_PREFIX * "_rep(\\d+)\\.csv\$"),
        save_path = "simulation_results/CA_paired_comparisons_multirep$(OUTPUT_TAG).csv")

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

    # Define the same comparisons as pairwise_paired_tests
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
        # Within-replicate mean differences
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

    # Print
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

# ============================================================================
# REWARD ANALYSIS
# ============================================================================

function print_top_percentile_differences(optimal_rewards, nearest_rewards; nsteps=1000, top_n=10)
    percentiles = range(0, stop=100, length=nsteps)
    results = []

    for (i, p) in enumerate(percentiles)
        opt_q = quantile(optimal_rewards, p/100)
        near_q = quantile(nearest_rewards, p/100)
        diff = opt_q - near_q
        push!(results, (percentile=p, diff=diff, opt_q=opt_q, near_q=near_q))
    end

    # Sort by largest difference (descending)
    sorted = sort(results, by=x->x.diff, rev=true)

    println("="^70)
    println("TOP $top_n PERCENTILE IMPROVEMENTS (Optimal vs Nearest)")
    println("="^70)
    println(rpad("Percentile", 12), rpad("Optimal", 10), rpad("Nearest", 10), rpad("Diff", 10))
    println("-"^50)
    for i in 1:top_n
        row = sorted[i]
        println(
            rpad(string(round(row.percentile, digits=2)), 12),
            rpad(string(round(row.opt_q, digits=4)), 10),
            rpad(string(round(row.near_q, digits=4)), 10),
            rpad(string(round(row.diff, digits=4)), 10)
        )
    end
    println()
    return sorted[1:top_n]
end

function analyze_rewards(optimal_rewards, nearest_rewards, h1_rewards=nothing, h2_rewards=nothing)
    println("="^70)
    println("REWARD ANALYSIS")
    println("="^70)

    policies = ["Optimal Policy", "Nearest Hospital"]
    reward_arrays = [optimal_rewards, nearest_rewards]

    if h1_rewards !== nothing
        push!(policies, "Heuristic 1 (CSC)")
        push!(reward_arrays, h1_rewards)
    end
    if h2_rewards !== nothing
        push!(policies, "Heuristic 2 (Any)")
        push!(reward_arrays, h2_rewards)
    end

    println("Summary Statistics:")
    println("Policy" * " "^20 * "Mean" * " "^8 * "Median" * " "^6 * "Std Dev" * " "^4 * "Std Error")
    println("-"^70)

    for (i, policy) in enumerate(policies)
        rewards = reward_arrays[i]
        avg = mean(rewards)
        med = median(rewards)
        std_dev = std(rewards)
        stderr = std_dev / sqrt(length(rewards))

        println(rpad(policy, 25) *
                rpad(string(round(avg, digits=4)), 12) *
                rpad(string(round(med, digits=4)), 12) *
                rpad(string(round(std_dev, digits=4)), 12) *
                string(round(stderr, digits=4)))
    end
    println()

    return reward_arrays
end

# ============================================================================
# REWARD AGREEMENTS
# ============================================================================

function analyze_reward_agreements(df)
    println("="^70)
    println("REWARD AGREEMENT ANALYSIS")
    println("="^70)

    n = nrow(df)

    # Pairwise comparisons
    opt_near = sum(df.optimal_action_reward .== df.nearest_hospital_reward)
    opt_h1 = hasproperty(df, :heuristic_1_reward) ? sum(df.optimal_action_reward .== df.heuristic_1_reward) : nothing
    opt_h2 = hasproperty(df, :heuristic_2_reward) ? sum(df.optimal_action_reward .== df.heuristic_2_reward) : nothing
    h1_h2 = (hasproperty(df, :heuristic_1_reward) && hasproperty(df, :heuristic_2_reward)) ? sum(df.heuristic_1_reward .== df.heuristic_2_reward) : nothing

    println("Agreements:")
    println("  Optimal vs Nearest:     $opt_near / $n ($(round(100*opt_near/n, digits=1))%)")
    if opt_h1 !== nothing
        println("  Optimal vs Heuristic 1: $opt_h1 / $n ($(round(100*opt_h1/n, digits=1))%)")
    end
    if opt_h2 !== nothing
        println("  Optimal vs Heuristic 2: $opt_h2 / $n ($(round(100*opt_h2/n, digits=1))%)")
    end
    if h1_h2 !== nothing
        println("  Heuristic 1 vs 2:       $h1_h2 / $n ($(round(100*h1_h2/n, digits=1))%)")
    end
    println()

    return Dict(
        "opt_near" => opt_near,
        "opt_h1" => opt_h1,
        "opt_h2" => opt_h2,
        "h1_h2" => h1_h2
    )
end

# ============================================================================
# TRAVEL TIMES
# ============================================================================

function analyze_travel_times(df)
    println("="^70)
    println("TRAVEL TIME ANALYSIS")
    println("="^70)

    # Update field names to match results_to_dataframe output!
    travel_time_fields = [
        (:travel_time_optimal, "Optimal"),
        (:travel_time_nh, "Nearest Hospital"),
        (:travel_time_h1, "Heuristic 1"),
        (:travel_time_h2, "Heuristic 2")
    ]

    println("Policy" * " "^17 * "Mean" * " "^8 * "Median" * " "^6 * "Std Dev" * " "^4 * "Std Error")
    println("-"^70)

    for (col, label) in travel_time_fields
        if hasproperty(df, col)
            vals = df[!, col]
            avg = mean(vals)
            med = median(vals)
            stddev = std(vals)
            stderr = stddev / sqrt(length(vals))
            println(rpad(label, 22) *
                    rpad(string(round(avg, digits=2)), 12) *
                    rpad(string(round(med, digits=2)), 12) *
                    rpad(string(round(stddev, digits=2)), 12) *
                    string(round(stderr, digits=2)))
        end
    end
    println()
end

# ============================================================================
# EXTREME CASE ANALYSIS
# ============================================================================

function analyze_extreme_cases(df)
    println("="^70)
    println("EXTREME CASE ANALYSIS")
    println("="^70)

    improvement = df.optimal_action_reward .- df.nearest_hospital_reward
    idx = argmax(improvement)
    max_improvement = improvement[idx]

    println("Maximum Reward Improvement Case:")
    println("  Improvement:           $(round(max_improvement, digits=4))")
    println("  Row index:             $idx")
    println("  Optimal reward:        $(round(df.optimal_action_reward[idx], digits=4))")
    println("  Nearest reward:        $(round(df.nearest_hospital_reward[idx], digits=4))")
    if hasproperty(df, :heuristic_1_reward)
        println("  Heuristic 1 reward:    $(round(df.heuristic_1_reward[idx], digits=4))")
    end
    if hasproperty(df, :heuristic_2_reward)
        println("  Heuristic 2 reward:    $(round(df.heuristic_2_reward[idx], digits=4))")
    end
    println()
    return idx
end

# ============================================================================
# OPTIMAL VS. HL
# ============================================================================

function print_cases_optimal_vs_h1_diff(df; save_csv::Bool=true)
    println("="^70)
    println("OPTIMAL vs HEURISTIC 1 ROUTING: DIFFERENT CASES")
    println("="^70)

    if !("optimal_action" in names(df)) || !("heuristic_1_action" in names(df))
        println("  Hospital routing columns not present. Skipping comparison.")
        return
    end

    # Identify rows where hospital routed to is different
    diff_idxs = findall(df.optimal_action .!= df.heuristic_1_action)
    n_diff = length(diff_idxs)
    n_total = nrow(df)

    println("Found $n_diff / $n_total cases where Optimal ≠ Heuristic 1.")

    same_reward_idxs = filter(i -> abs(df.optimal_action_reward[i] - df.heuristic_1_reward[i]) < 1e-4, diff_idxs)
    n_same = length(same_reward_idxs)

    println("  → $n_same of these have identical reward values.")

    # Calc extra travel time for same-reward cases
    Δtravel = df.travel_time_h1[same_reward_idxs] .- df.travel_time_optimal[same_reward_idxs]
    avg_extra_time = mean(Δtravel)
    median_extra_time = median(Δtravel)
    std_err_extra_time = std(Δtravel) / sqrt(n_same)
    println("  → Average extra travel time for same-reward cases: $(round(avg_extra_time, digits=2)) ± $(round(std_err_extra_time, digits=2)) minutes")
    println("  → Median extra travel time for same-reward cases: $(round(median_extra_time, digits=2)) minutes")
    println("  → Standard error of extra travel time: $(round(std_err_extra_time, digits=2)) minutes")

    # Print maximum extra travel time
    max_extra_time = maximum(Δtravel)
    max_idx = same_reward_idxs[argmax(Δtravel)]
    println("  → Maximum extra travel time: $(round(max_extra_time, digits=2)) minutes at index $max_idx")

    # Print info on max case
    println("Details for maximum extra travel time case:")
    println("Case $max_idx:")
    println("  Start State: lat=$(round(df.start_lat[max_idx], digits=4)), lon=$(round(df.start_lon[max_idx], digits=4)), t_onset=$(round(df.t_onset[max_idx], digits=2)), stroke_type=$(df.stroke_type[max_idx])")
    println("  Optimal:    $(df.optimal_action[max_idx]), travel_time=$(round(df.travel_time_optimal[max_idx], digits=2)), reward=$(round(df.optimal_action_reward[max_idx], digits=4))")
    println("  Heuristic1: $(df.heuristic_1_action[max_idx]), travel_time=$(round(df.travel_time_h1[max_idx], digits=2)), reward=$(round(df.heuristic_1_reward[max_idx], digits=4))")
    println()

    # Unique hospitals routed to (entire dataset)
    unique_opt_hospitals = unique(df.optimal_action)
    unique_h1_hospitals = unique(df.heuristic_1_action)

    println("Hospital routing across all cases:")
    println("  → Unique hospitals under Optimal Policy:    $(length(unique_opt_hospitals))")
    println("  → Unique hospitals under Heuristic 1:       $(length(unique_h1_hospitals))")

    # Unique hospitals in divergent cases
    unique_opt_diff = unique(df.optimal_action[diff_idxs])
    unique_h1_diff = unique(df.heuristic_1_action[diff_idxs])

    println("Hospital routing in divergent cases only:")
    println("  → Unique hospitals under Optimal (diff only):    $(length(unique_opt_diff))")
    println("  → Unique hospitals under Heuristic 1 (diff only): $(length(unique_h1_diff))")

    # Save the filtered divergent DataFrame
    if save_csv
        df_diff = df[diff_idxs, :]
        CSV.write("simulation_results/CA_divergent_optimal_vs_h1.csv", df_diff)
        println("✓ Divergent cases saved to simulation_results/CA_divergent_optimal_vs_h1.csv")
    end
end

# ============================================================================
# VISUALIZATION FUNCTIONS 
# ============================================================================

function create_histogram_comparison(df)
    println("Creating histogram comparison...")

    optimal = df.optimal_action_reward
    nearest = df.nearest_hospital_reward

    # Compute statistics for annotations
    mean_opt = mean(optimal)
    median_opt = median(optimal)
    mean_near = mean(nearest)
    median_near = median(nearest)

    # Create histogram
    darker_gray = RGB(0.4, 0.4, 0.4)
    histogram(nearest, bins=40, alpha=0.35, color=darker_gray, label="Nearest Hospital",
        linewidth=0, size=(800, 400), legend=:outertopright)
    histogram!(optimal, bins=40, alpha=0.5, color=:lightblue, label="Optimal Policy", linewidth=0)

    # Add mean and median lines
    vline!([mean_near], color=:black, linestyle=:dash, linewidth=2, label="Mean (Nearest)")
    vline!([mean_opt], color=:blue, linestyle=:dash, linewidth=2, label="Mean (Optimal)")
    vline!([median_near], color=:black, linestyle=:dot, linewidth=2, label="Median (Nearest)")
    vline!([median_opt], color=:blue, linestyle=:dot, linewidth=2, label="Median (Optimal)")

    xlabel!("Probability of Good Outcome")
    ylabel!("Number of Patients")
    title!("Patient Outcome Probabilities by Policy")

    plot!(
        legend=:topright,
        legendfontsize=8,
        left_margin=10mm,
        right_margin=10mm,
        top_margin=5mm,
        bottom_margin=10mm,
        framestyle=:box,
        guidefont=font(10, "Times New Roman"),
        tickfont=font(9, "Times New Roman"),
        legendfont=font(9, "Times New Roman")
    )

    savefig("simulation_results/CA_optimal_vs_nearest_histogram.pdf")
    println("✓ Saved histogram to simulation_results/CA_optimal_vs_nearest_histogram.pdf")
end

function create_visualizations(df, reward_arrays=nothing, policy_names=nothing)
    println("="^70)
    println("CREATING VISUALIZATIONS")
    println("="^70)

    create_histogram_comparison(df)

    println()
end

# ============================================================================
# MAIN ANALYSIS EXECUTION
# ============================================================================

function run_complete_analysis()
    println("Starting comprehensive stroke triage simulation analysis...")
    println()

    # Load data
    data_result = load_and_prepare_data()
    df = data_result[1]
    opt_rewards, near_rewards, h1_rewards, h2_rewards = data_result[2:5]

    # Primary analysis: paired t-test CIs for every policy comparison
    pairwise_paired_tests(df)

    # Subgroup analyses (equity-relevant breakdowns of the primary analysis):
    #   1) by stroke type   — does the MDP help LVO patients more than others?
    #   2) by region        — does the MDP help rural patients more than urban?
    if hasproperty(df, :stroke_type)
        subgroup_paired_tests(df, :stroke_type;
            label_for_print = "by stroke type",
            save_path = "simulation_results/CA_paired_comparisons_by_stroke_type$(OUTPUT_TAG).csv")
    end
    add_region_column!(df)
    if hasproperty(df, :region)
        subgroup_paired_tests(df, :region;
            label_for_print = "by region (rurality, proxied by nearest-hospital travel time)",
            save_path = "simulation_results/CA_paired_comparisons_by_region$(OUTPUT_TAG).csv")
    end

    # Multi-replicate aggregation (IEEE-style protocol). Skipped silently
    # if no replicate CSVs are present.
    try
        files = filter(f -> occursin(Regex("^" * INPUT_PREFIX * "_rep\\d+\\.csv\$"), f),
                       readdir("simulation_results"))
        if length(files) >= 2
            aggregate_replicates()
        end
    catch e
        @warn "Multi-replicate aggregation skipped: $e"
    end

    # Reward analysis
    reward_arrays = analyze_rewards(opt_rewards, near_rewards, h1_rewards, h2_rewards)
    policy_names = ["Optimal", "Nearest"]
    if h1_rewards !== nothing
        push!(policy_names, "Heuristic 1")
    end
    if h2_rewards !== nothing
        push!(policy_names, "Heuristic 2")
    end

    # Reward agreement analysis (counts how often rewards are exactly equal)
    analyze_reward_agreements(df)
    print_top_percentile_differences(opt_rewards, near_rewards)

    # Travel time analysis
    analyze_travel_times(df)

    # Extreme case analysis
    analyze_extreme_cases(df)

    # Visualizations
    create_visualizations(df, reward_arrays, policy_names)

    # Optimal vs Heuristic 1 case differences
    print_cases_optimal_vs_h1_diff(df)

    println("="^70)
    println("ANALYSIS COMPLETE")
    println("="^70)

    return df
end

# ============================================================================
# EXECUTION
# ============================================================================

df = run_complete_analysis()