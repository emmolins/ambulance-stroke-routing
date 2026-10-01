#=
File: decision_tree_sweep.jl
================================================================================
Decision-tree depth sweep — distilling the MDP-optimal policy into an
interpretable, EMS-deployable rule.

This script positions the decision-tree distillation as a core contribution
of the paper rather than an auxiliary analysis. It sweeps tree depth over a
range, evaluates each tree's outcome quality against the MDP-optimal policy
on a held-out test set, and identifies the smallest tree that recovers most
of the optimal policy's value.

WHAT THIS DOES
--------------
1. Loads the already-generated training data (13 features per patient,
   MDP-optimal hospital TYPE as the label) from
   `decision_tree_output/training_data_detailed.csv`.
2. Trains DecisionTreeClassifier at depths [1, 2, 3, 4, 5, 6, 8, 10, ∞].
3. Picks N_TEST patient locations held out from training, and for each one:
     a. Builds the same 13-feature vector.
     b. For every tree depth: predicts the hospital TYPE, routes the patient
        to the nearest reachable hospital of that type, computes the reward.
     c. Also computes the four reference policies on the same patient:
        MDP optimal (depth-2 forward search), Nearest, Heuristic 1, Heuristic 2.
4. Aggregates per-depth metrics:
     * mean reward
     * mean outcome loss vs MDP-optimal (= mean(MDP_reward − tree_reward))
     * action-type accuracy (% of patients where tree's predicted type matches
       the MDP-optimal hospital's type)
     * tree complexity (#leaves)
5. Saves:
     `decision_tree_output/sweep_summary.csv`   — one row per depth + baselines
     `decision_tree_output/sweep_perpatient.csv` — long-form per-patient table
     `decision_tree_output/sweep_tradeoff.pdf`  — depth vs outcome-loss figure

REQUIREMENTS
------------
* ORS must be running (`docker compose up -d` in ~/ors).
* `decision_tree_output/training_data_detailed.csv` must exist
  (run `decision_tree_build.jl` once first if it doesn't).
================================================================================
=#

using CSV
using DataFrames
using DecisionTree
using Random
using Statistics
using StatsBase
using Plots
using ProgressMeter

include("../src/CA_STPMDP_ORS.jl")

# ============================================================================
# Configuration
# ============================================================================
SEED         = 2025
N_TEST       = 200                       # patients per sweep run (kept moderate; ORS-bound)
DEPTHS       = [1, 2, 3, 4, 5, 6, 8, 10, -1]   # -1 = unconstrained (set to nothing in DT)
OUTPUT_DIR   = "decision_tree_output"
TRAIN_CSV    = joinpath(OUTPUT_DIR, "training_data_detailed.csv")
POINTS_CSV   = "sampled_points/CA_points.csv"

Random.seed!(SEED)

# Hospital-type labels (same order as decision_tree_build.jl)
TYPE_LABELS  = ["Route_CSC", "Route_PSC", "Route_Clinic"]
const LBL_CSC = 1
const LBL_PSC = 2
const LBL_CLINIC = 3

# ============================================================================
# 1. Load training data
# ============================================================================
println("="^78)
println("DECISION-TREE DEPTH SWEEP")
println("="^78)
println("\n[1/5] Loading training data from $TRAIN_CSV ...")

train_df = CSV.read(TRAIN_CSV, DataFrame)
println("  loaded $(nrow(train_df)) training samples")

# Build feature matrix in the same column order as decision_tree_build.jl uses.
FEATURE_NAMES = [
    "Time to CSC", "Time to PSC", "Time to Clinic", "Time since onset",
    "CSC Reachable", "PSC Reachable", "Clinic Reachable",
    "Diff CSC-PSC", "Diff CSC-Clinic", "Diff PSC-Clinic",
    "Ratio CSC/PSC", "Ratio CSC/Clinic", "Ratio PSC/Clinic",
]
train_features = Matrix(hcat(
    train_df.Time_to_CSC_min,
    train_df.Time_to_PSC_min,
    train_df.Time_to_Clinic_min,
    train_df.Time_since_onset_min,
    Float64.(train_df.CSC_Reachable),
    Float64.(train_df.PSC_Reachable),
    Float64.(train_df.Clinic_Reachable),
    train_df.Diff_CSC_PSC,
    train_df.Diff_CSC_Clinic,
    train_df.Diff_PSC_Clinic,
    train_df.Ratio_CSC_PSC,
    train_df.Ratio_CSC_Clinic,
    train_df.Ratio_PSC_Clinic,
))
train_labels = train_df.Predicted_Label
@assert length(train_labels) == size(train_features, 1)

# ============================================================================
# 2. Train trees at all depths
# ============================================================================
println("\n[2/5] Training decision trees at depths $(DEPTHS) ...")

models = Dict{Int, Any}()
for d in DEPTHS
    md = d == -1 ? -1 : d                       # DecisionTree.jl uses -1 for unlimited
    m = DecisionTreeClassifier(max_depth=md)
    DecisionTree.fit!(m, train_features, train_labels)
    models[d] = m
    println("  depth=$(d == -1 ? "∞" : d): $(DecisionTree.length(m.root)) total nodes")
end

# ============================================================================
# 3. Held-out test set: sample N_TEST locations that weren't in training
# ============================================================================
println("\n[3/5] Building held-out test set of $N_TEST patients ...")

train_used_set = Set(zip(train_df.Time_to_CSC_min, train_df.Time_to_PSC_min))  # rough match
all_points = [(row.Latitude, row.Longitude) for row in CSV.File(POINTS_CSV)]
shuffle!(all_points)

mdp = StrokeMDP()

# Helper: build the 13-feature vector for a candidate test patient
function build_features(s::PatientState)
    csc_h = find_nearest_CSC(mdp, s.loc)
    psc_h = find_nearest_PSC(mdp, s.loc)
    cli_h = find_nearest_clinic(mdp, s.loc)
    csc_r = csc_h !== nothing
    psc_r = psc_h !== nothing
    cli_r = cli_h !== nothing
    t_csc = csc_r ? calculate_travel_time(s.loc, csc_h) : 1e8
    t_psc = psc_r ? calculate_travel_time(s.loc, psc_h) : 1e8
    t_cli = cli_r ? calculate_travel_time(s.loc, cli_h) : 1e8
    # If travel-time call returned nothing (extreme), fall back
    t_csc === nothing && (t_csc = 1e8; csc_r = false)
    t_psc === nothing && (t_psc = 1e8; psc_r = false)
    t_cli === nothing && (t_cli = 1e8; cli_r = false)
    diff_cp = (csc_r && psc_r) ? abs(t_csc - t_psc) : 0.0
    diff_cc = (csc_r && cli_r) ? abs(t_csc - t_cli) : 0.0
    diff_pc = (psc_r && cli_r) ? abs(t_psc - t_cli) : 0.0
    ratio_cp = (csc_r && psc_r && t_psc > 0) ? t_csc / t_psc : 1.0
    ratio_cc = (csc_r && cli_r && t_cli > 0) ? t_csc / t_cli : 1.0
    ratio_pc = (psc_r && cli_r && t_cli > 0) ? t_psc / t_cli : 1.0
    feats = Float64[
        t_csc, t_psc, t_cli, s.t_onset,
        Float64(csc_r), Float64(psc_r), Float64(cli_r),
        diff_cp, diff_cc, diff_pc,
        ratio_cp, ratio_cc, ratio_pc,
    ]
    return feats, (csc_h, psc_h, cli_h)
end

# Helper: reward of routing to a given destination (nothing = no route)
function reward_of_route(s::PatientState, dest::Union{Location, Nothing})
    dest === nothing && return missing
    action_str = "ROUTE_" * dest.name
    a = string_to_enum(action_str)
    sp = rand(transition(mdp, s, a))
    return reward(mdp, s, a, sp), action_str
end

# ============================================================================
# 4. Evaluate every tree + every baseline on the test patients
# ============================================================================
println("\n[4/5] Evaluating policies on $N_TEST test patients ...")

# Long-form record: one row per (patient, policy)
per_patient = DataFrame(
    sample_id   = Int[],
    policy      = String[],
    action      = String[],
    reward      = Union{Missing, Float64}[],
    mdp_reward  = Float64[],
)

progress = Progress(N_TEST; desc = "patients ")
idx = 0
collected = 0
while collected < N_TEST && idx < length(all_points)
    idx += 1
    latlon = all_points[idx]
    s = PatientState(
        Location("FIELD$collected", latlon, -1, FIELD),
        30 + rand() * 240,
        UNKNOWN,
        sample_stroke_type(mdp),
    )

    feats, (csc_h, psc_h, cli_h) = build_features(s)

    # MDP-optimal baseline (always run first; if it fails, skip patient entirely)
    mdp_action = best_action(mdp, s, 2)
    mdp_action === nothing && continue
    mdp_action_str = enum_to_string(mdp_action)
    sp_mdp = rand(transition(mdp, s, mdp_action))
    mdp_reward = reward(mdp, s, mdp_action, sp_mdp)

    collected += 1
    sid = collected

    # MDP row (used as both a comparator and the reference for outcome loss)
    push!(per_patient, (sid, "MDP_optimal", mdp_action_str, mdp_reward, mdp_reward))

    # Tree rows — one per depth
    type_to_dest = (csc_h, psc_h, cli_h)
    for d in DEPTHS
        pred = DecisionTree.predict(models[d], feats)
        dest = type_to_dest[pred]
        r, a_str = reward_of_route(s, dest)
        push!(per_patient, (sid, "tree_depth_$(d == -1 ? "inf" : d)",
                             a_str === nothing ? "no_reachable_$(TYPE_LABELS[pred])" : a_str,
                             r, mdp_reward))
    end

    # Existing baseline policies (Nearest / Heur1 / Heur2) — reuse the fallback-chain
    # versions in CA_STPMDP_ORS.jl so this aligns with the simulation comparison.
    function policy_eval(name, action_str_or_nothing)
        if action_str_or_nothing === nothing
            push!(per_patient, (sid, name, "no_valid_action", missing, mdp_reward))
            return
        end
        a = string_to_enum(action_str_or_nothing)
        sp = rand(transition(mdp, s, a))
        push!(per_patient, (sid, name, action_str_or_nothing,
                             reward(mdp, s, a, sp), mdp_reward))
    end
    policy_eval("Nearest",     current_practice_action(mdp, s))
    policy_eval("Heuristic_1", heuristic_1_action(mdp, s))
    policy_eval("Heuristic_2", heuristic_2_action(mdp, s))

    next!(progress)
end
finish!(progress)

println("  evaluated $(collected) patients (× $(length(DEPTHS) + 4) policies each)")

mkpath(OUTPUT_DIR)
CSV.write(joinpath(OUTPUT_DIR, "sweep_perpatient.csv"), per_patient)

# ============================================================================
# 5. Aggregate per-policy summary
# ============================================================================
println("\n[5/5] Aggregating per-policy summary ...")

# Per-policy aggregates
function policy_summary(df, policy_name)
    sub = df[df.policy .== policy_name, :]
    sub = sub[.!ismissing.(sub.reward), :]
    nrow(sub) == 0 && return nothing
    rewards    = Float64.(sub.reward)
    mdp_rew    = sub.mdp_reward
    diffs      = mdp_rew .- rewards          # outcome loss vs MDP
    return (
        policy           = policy_name,
        n_evaluated      = nrow(sub),
        mean_reward      = mean(rewards),
        outcome_loss_mean = mean(diffs),
        outcome_loss_se  = std(diffs) / sqrt(length(diffs)),
        recovered_pct    = 100 * (1 - mean(diffs) / max(mean(mdp_rew), 1e-9)),
    )
end

all_policies = ["MDP_optimal";
                ["tree_depth_$(d == -1 ? "inf" : d)" for d in DEPTHS];
                ["Nearest", "Heuristic_1", "Heuristic_2"]]

summary_rows = NamedTuple[]
for p in all_policies
    s = policy_summary(per_patient, p)
    s !== nothing && push!(summary_rows, s)
end
summary_df = DataFrame(summary_rows)

# Attach tree-complexity columns for the depth rows (n_nodes / max_depth)
summary_df.depth        = [startswith(r.policy, "tree_depth_") ?
                             (r.policy == "tree_depth_inf" ? -1 :
                              parse(Int, replace(r.policy, "tree_depth_" => ""))) :
                             missing for r in eachrow(summary_df)]
summary_df.n_nodes      = [startswith(r.policy, "tree_depth_") ?
                             DecisionTree.length(models[r.depth].root) : missing
                           for r in eachrow(summary_df)]

CSV.write(joinpath(OUTPUT_DIR, "sweep_summary.csv"), summary_df)

# Print a readable table
println()
println("="^110)
println("DECISION-TREE DEPTH SWEEP — summary")
println("="^110)
println("Policy                       n       mean reward    outcome loss    % of MDP recovered    nodes")
println("-"^110)
for r in eachrow(summary_df)
    printstyled(rpad(r.policy, 26), color=:default)
    print(rpad(string(r.n_evaluated), 8))
    print(rpad(string(round(r.mean_reward, digits=4)), 14))
    print(rpad("$(round(r.outcome_loss_mean, digits=5)) ± $(round(r.outcome_loss_se, digits=5))", 17))
    print(rpad("$(round(r.recovered_pct, digits=2))%", 22))
    println(ismissing(r.n_nodes) ? "—" : string(r.n_nodes))
end
println()
println("✓ Wrote $(joinpath(OUTPUT_DIR, "sweep_summary.csv"))")
println("✓ Wrote $(joinpath(OUTPUT_DIR, "sweep_perpatient.csv"))")

# ============================================================================
# 6. Tradeoff figure: depth vs outcome loss (+ baselines as horizontal lines)
# ============================================================================
println("\nGenerating tradeoff plot ...")

tree_rows = summary_df[.!ismissing.(summary_df.depth), :]
sort!(tree_rows, :depth)
xs = [r.depth == -1 ? 20 : r.depth for r in eachrow(tree_rows)]     # ∞ shown at x=20
ys = tree_rows.outcome_loss_mean
errs = 1.96 .* tree_rows.outcome_loss_se

plt = plot(
    xs, ys, yerror = errs,
    seriestype = :line, marker = :circle, lw = 2, ms = 6,
    label = "Decision tree at depth d",
    xlabel = "Tree depth",
    ylabel = "Outcome loss vs MDP-optimal\nmean(P(good)_MDP − P(good)_policy)",
    title  = "Interpretability–accuracy tradeoff",
    legend = :topright,
    framestyle = :box,
    size = (760, 460),
    dpi = 140,
)

# Overlay baseline policies as horizontal lines
for (name, color, style) in [
    ("Nearest",     :black,  :dash),
    ("Heuristic_1", :green,  :dashdot),
    ("Heuristic_2", :orange, :dot),
]
    row = summary_df[summary_df.policy .== name, :]
    nrow(row) == 0 && continue
    hline!([row.outcome_loss_mean[1]], color=color, lw=2, ls=style, label=name)
end

savefig(plt, joinpath(OUTPUT_DIR, "sweep_tradeoff.pdf"))
println("✓ Wrote $(joinpath(OUTPUT_DIR, "sweep_tradeoff.pdf"))")

# ============================================================================
# 7. Pick & report the "sweet spot": smallest tree within 1% of MDP-optimal
# ============================================================================
println()
target_pct = 99.0
sweet = nothing
for r in eachrow(sort(tree_rows, :depth))
    if r.recovered_pct >= target_pct
        sweet = r
        break
    end
end

if sweet !== nothing
    println("="^78)
    println("RECOMMENDED DEPLOYABLE POLICY")
    println("="^78)
    println("Smallest tree that recovers ≥ $target_pct% of MDP-optimal expected outcome:")
    println("  depth      = $(sweet.depth == -1 ? "∞" : sweet.depth)")
    println("  # nodes    = $(sweet.n_nodes)")
    println("  mean reward      = $(round(sweet.mean_reward, digits=4))")
    println("  outcome loss     = $(round(sweet.outcome_loss_mean, digits=5)) ± $(round(sweet.outcome_loss_se, digits=5))")
    println("  % MDP recovered  = $(round(sweet.recovered_pct, digits=2))%")
else
    println("No tree depth in the sweep reached $(target_pct)% of MDP-optimal.")
    println("Consider raising max depth or N_TEST, or revisiting the feature set.")
end

println("\nDONE.")
