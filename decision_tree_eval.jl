#=
File: decision_tree_eval.jl
----------------------------
This script evaluates a trained decision tree stroke triage policy on 
a held-out test set.For each test point, it simulates onset time and 
stroke type, computes routing features, andcompares the decision tree's 
reward to the optimal policy using the MDP. Outputs include summary 
statistics (mean, median, std, stderr) and a CSV with all test results,
saved in the decision_tree_output folder for further analysis.
=#

using CSV
using DataFrames
using DecisionTree
using JLD2
using Statistics
using ProgressBars
using StatsBase
import Base.Filesystem: mkpath
using Random
SEED = 2025  
Random.seed!(SEED)

const OUTPUT_DIR = "decision_tree_output"
mkpath(OUTPUT_DIR)

include("CA_STPMDP_ORS.jl")
include("decision_tree_utils.jl")

# Load the trained model and feature information
@load joinpath(OUTPUT_DIR, "decision_tree_model.jld2") model feature_names action_labels

println("Loaded decision tree model with features: ", feature_names)
println("Action labels: ", action_labels)

# Load test points
test_df = CSV.File(joinpath(OUTPUT_DIR, "decision_tree_test.csv")) |> DataFrame
test_points = [(row.Latitude, row.Longitude) for row in eachrow(test_df)]
println("Loaded $(length(test_points)) test points")

# Slice to use only the first n test points
N_TEST = 1000
test_points = test_points[1:min(N_TEST, length(test_points))]

println("Evaluating on $(length(test_points)) test points")
println("Loaded $(length(test_points)) test points")

# Initialize MDP 
mdp = StrokeMDP()

# Set parameters for test evaluation
N_TEST = length(test_points)
test_results = DataFrame(
    Sample_ID = 1:N_TEST,
    Latitude = [lat for (lat, lon) in test_points],
    Longitude = [lon for (lat, lon) in test_points],
    Onset_Time = zeros(Float64, N_TEST),
    Stroke_Type = fill("", N_TEST),
    t_CSC = zeros(Float64, N_TEST),
    t_PSC = zeros(Float64, N_TEST),
    t_Clinic = zeros(Float64, N_TEST),
    Features = [[] for _ in 1:N_TEST],
    DT_Action = fill("", N_TEST),
    DT_Reward = zeros(Float64, N_TEST),
    Optimal_Action = fill("", N_TEST),
    Optimal_Reward = zeros(Float64, N_TEST)
)

valid_samples = 0
attempts = 0
max_attempts = 10 * N_TEST
used_indices = Set{Int}()


test_results = DataFrame(
    Sample_ID = Int[],
    Latitude = Float64[],
    Longitude = Float64[],
    Onset_Time = Float64[],
    Stroke_Type = String[],
    t_CSC = Float64[],
    t_PSC = Float64[],
    t_Clinic = Float64[],
    Features = Vector{Any}[],
    DT_Action = String[],
    DT_Reward = Float64[],
    Optimal_Action = String[],
    Optimal_Reward = Float64[]
)

progress = ProgressBar(1:N_TEST)

while valid_samples < N_TEST && attempts < max_attempts
    global attempts, used_indices, valid_samples
    idx = rand(1:length(test_points))
    if idx in used_indices
        attempts += 1
        continue
    end
    push!(used_indices, idx)

    loc = Location("FIELD1", test_points[idx], -1, FIELD)
    t_onset = rand() * 270
    st_type = sample_stroke_type(mdp)
    s = PatientState(loc, t_onset, UNKNOWN, st_type)

    # Declare these as local to avoid soft scope warning
    local csc_reachable, psc_reachable, clinic_reachable

    csc_hosp, t_nearest_CSC = safe_find_nearest_hospital(mdp, loc, find_nearest_CSC)
    psc_hosp, t_nearest_PSC = safe_find_nearest_hospital(mdp, loc, find_nearest_PSC)
    clinic_hosp, t_nearest_clinic = safe_find_nearest_hospital(mdp, loc, find_nearest_clinic)

    csc_reachable = t_nearest_CSC !== nothing
    psc_reachable = t_nearest_PSC !== nothing
    clinic_reachable = t_nearest_clinic !== nothing

    # Replace unreachable with large number
    t_csc = csc_reachable ? t_nearest_CSC : 1.0e8
    t_psc = psc_reachable ? t_nearest_PSC : 1.0e8
    t_clinic = clinic_reachable ? t_nearest_clinic : 1.0e8

    # Differences
    diff_CSC_PSC = (csc_reachable && psc_reachable) ? abs(t_csc - t_psc) : 0.0
    diff_CSC_Clinic = (csc_reachable && clinic_reachable) ? abs(t_csc - t_clinic) : 0.0
    diff_PSC_Clinic = (psc_reachable && clinic_reachable) ? abs(t_psc - t_clinic) : 0.0

    # Ratios
    ratio_CSC_PSC = (csc_reachable && psc_reachable && t_psc > 0) ? t_csc / t_psc : 1.0
    ratio_CSC_Clinic = (csc_reachable && clinic_reachable && t_clinic > 0) ? t_csc / t_clinic : 1.0
    ratio_PSC_Clinic = (psc_reachable && clinic_reachable && t_clinic > 0) ? t_psc / t_clinic : 1.0

    # Build feature vector with 13 features in same order as training
    test_feats = [
        t_csc, t_psc, t_clinic, t_onset,
        Float64(csc_reachable), Float64(psc_reachable), Float64(clinic_reachable),
        diff_CSC_PSC, diff_CSC_Clinic, diff_PSC_Clinic,
        ratio_CSC_PSC, ratio_CSC_Clinic, ratio_PSC_Clinic
    ]

    dt_label = DecisionTree.predict(model, test_feats)
    dt_action = action_labels[dt_label]

    if dt_action == "Route_CSC"
        target_loc, _ = safe_find_nearest_hospital(mdp, s.loc, find_nearest_CSC)
    elseif dt_action == "Route_PSC"
        target_loc, _ = safe_find_nearest_hospital(mdp, s.loc, find_nearest_PSC)
    elseif dt_action == "Route_Clinic"
        target_loc, _ = safe_find_nearest_hospital(mdp, s.loc, find_nearest_clinic)
    else
        error("Unknown action label: $dt_action")
    end    
    
    if target_loc === nothing
        println("Warning: No reachable hospital of type $dt_action at location $(s.loc.latlon); skipping sample $(valid_samples+1).")
        attempts += 1
        continue  # Skip to next sample
    end
    
    dt_action_str = "ROUTE_" * target_loc.name
    enum_dt_action = string_to_enum(dt_action_str)
    

    dt_next = rand(transition(mdp, s, enum_dt_action))
    dt_reward = reward(mdp, s, enum_dt_action, dt_next)

    optimal_action = best_action(mdp, s, 2)
    optimal_action_str = enum_to_string(optimal_action)
    optimal_next = rand(transition(mdp, s, optimal_action))
    optimal_reward = reward(mdp, s, optimal_action, optimal_next)

    push!(test_results, (
        valid_samples + 1,
        test_points[idx][1],
        test_points[idx][2],
        t_onset,
        string(st_type),
        t_csc,
        t_psc,
        t_clinic,
        test_feats,
        dt_action,
        dt_reward,
        optimal_action_str,
        optimal_reward
    ))

    valid_samples += 1
    attempts += 1
end

if valid_samples < N_TEST
    println("Warning: Only $(valid_samples) valid test points found after $attempts attempts.")
end

println("Test evaluation complete on $(N_TEST) samples.")

# --- Summarize and print results ---
println("\nSUMMARY STATISTICS:")
mean_dt = mean(test_results.DT_Reward)
mean_opt = mean(test_results.Optimal_Reward)
median_dt = median(test_results.DT_Reward)
median_opt = median(test_results.Optimal_Reward)
std_dt = std(test_results.DT_Reward)
std_opt = std(test_results.Optimal_Reward)
stderr_dt = std_dt / sqrt(N_TEST)
stderr_opt = std_opt / sqrt(N_TEST)
println("Mean reward (Decision Tree):   ", round(mean_dt, digits=4))
println("Mean reward (Optimal Policy):  ", round(mean_opt, digits=4))
println("Median reward (Decision Tree): ", round(median_dt, digits=4))
println("Median reward (Optimal Policy):", round(median_opt, digits=4))

println("Std Dev (Decision Tree):        ", round(std_dt, digits=4))
println("Std Dev (Optimal Policy):       ", round(std_opt, digits=4))
println("Std Error (Decision Tree):      ", round(stderr_dt, digits=4))
println("Std Error (Optimal Policy):     ", round(stderr_opt, digits=4))

# Save detailed results to CSV for further analysis
CSV.write(joinpath(OUTPUT_DIR, "decision_tree_test_results.csv"), test_results)
println("✓ Results saved as $(joinpath(OUTPUT_DIR, "decision_tree_test_results.csv"))")