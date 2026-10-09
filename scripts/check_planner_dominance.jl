# Sanity check for the terminal-reward planner: on planning reward the MDP's
# first destination must never score below any comparator's, and the per-step
# reward on the first leg must be zero.
#   julia --project=. scripts/check_planner_dominance.jl            (CA, 200 patients)
#   REGION=ri N=100 julia --project=. scripts/check_planner_dominance.jl
using CSV, DataFrames, Random
const REGION = lowercase(get(ENV, "REGION", "ca"))
const N      = parse(Int, get(ENV, "N", "200"))
include(REGION == "ri" ? "../src/RI_STPMDP_ORS.jl" : "../src/CA_STPMDP_ORS.jl")
Random.seed!(2026)
mdp = StrokeMDP()
pts = CSV.read(REGION == "ri" ? "sampled_points/RI_points.csv" : "sampled_points/CA_points.csv", DataFrame)

worst = 0.0; n_ok = 0; n_tie = 0; n_better = 0
for i in 1:N
    row = pts[rand(1:nrow(pts)), :]
    s = PatientState(Location("FIELD$i", (row.Latitude, row.Longitude), -1, FIELD),
                     sample_onset_time(), UNKNOWN, sample_stroke_type(mdp))
    a_mdp = best_action(mdp, s, 2)
    a_mdp === nothing && continue
    sp = rand(transition(mdp, s, a_mdp))
    @assert reward(mdp, s, a_mdp, sp) == 0.0 "first-leg reward must be zero"
    v_mdp = action_value(mdp, s, a_mdp, sp)
    for a_str in (current_practice_action(mdp, s), heuristic_1_action(mdp, s), heuristic_2_action(mdp, s))
        a_str === nothing && continue
        a = string_to_enum(a_str)
        v = action_value(mdp, s, a)
        global worst = min(worst, v_mdp - v)
        v_mdp - v > 1e-9 ? (global n_better += 1) : (global n_tie += 1)
    end
    global n_ok += 1
end
println("patients checked: $n_ok   comparisons: better $n_better, tie $n_tie")
println("worst MDP - comparator on planning reward: $worst  (must be >= -1e-9)")
println(worst >= -1e-9 ? "PASS" : "FAIL")
