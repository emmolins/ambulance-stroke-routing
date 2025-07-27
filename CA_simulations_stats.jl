#=
File: CA_simulations_stats.jl
----------------------------------------------------------------------
Performs comprehensive statistical analysis of stroke triage simulation 
results comparing optimal MDP-based ambulance routing to heuristic 
strategies across California. Includes reward comparisons, hypothesis 
testing, travel time evaluation, agreement analysis, and visualizations.
=#

using CSV, Statistics, Distributions, DataFrames, Measures, StatsPlots, Plots, StatsBase

# Set default plot styling
default(fontfamily="Times New Roman")

# Load and prep data for analysis
function load_and_prepare_data()
    println("="^70)
    println("DATA LOADING AND PREPARATION")
    println("="^70)

    # Load CSV data
    df = CSV.read("simulation_results/CA_simulation_results.csv", DataFrame)
    println("✓ Loaded CSV data with $(nrow(df)) observations")

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
# STATISTICAL HYPOTHESIS TESTING
# ============================================================================

function perform_hypothesis_tests(df)
    println("="^70)
    println("STATISTICAL HYPOTHESIS TESTING")
    println("="^70)

    optimal_rewards = df.optimal_action_reward
    nearest_rewards = df.nearest_hospital_reward

    # Paired t-test for optimal vs nearest hospital
    diffs = optimal_rewards .- nearest_rewards
    n = length(diffs)
    mean_diff = mean(diffs)
    std_diff = std(diffs)
    se = std_diff / sqrt(n)
    t_stat = mean_diff / se
    p_val = 2 * (1 - cdf(TDist(n - 1), abs(t_stat)))

    # 95% Confidence interval
    ci_low = mean_diff - quantile(TDist(n - 1), 0.975) * se
    ci_high = mean_diff + quantile(TDist(n - 1), 0.975) * se

    println("Paired t-test: Optimal vs Nearest Hospital Policy")
    println("  Sample size:           $n")
    println("  Mean difference:       $(round(mean_diff, digits=6))")
    println("  Standard deviation:    $(round(std_diff, digits=6))")
    println("  Standard error:        $(round(se, digits=6))")
    println("  t-statistic:           $(round(t_stat, digits=4))")
    println("  p-value:               $(round(p_val, digits=8))")
    println("  95% CI:                [$(round(ci_low, digits=6)), $(round(ci_high, digits=6))]")

    if p_val < 0.001
        println("  Result:                Highly significant (p < 0.001)")
    elseif p_val < 0.01
        println("  Result:                Significant (p < 0.01)")
    elseif p_val < 0.05
        println("  Result:                Significant (p < 0.05)")
    else
        println("  Result:                Not significant (p ≥ 0.05)")
    end
    println()

    return mean_diff, std_diff, se, t_stat, p_val, ci_low, ci_high
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

    # Statistical hypothesis testing
    perform_hypothesis_tests(df)

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