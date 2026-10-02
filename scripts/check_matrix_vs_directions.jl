# =============================================================================
# scripts/check_matrix_vs_directions.jl
#
# Verifies that travel times served by the ORS Matrix endpoint (used by the
# planner through cached_travel_time) agree with the per-pair Directions
# endpoint the code used before 3 Oct 2026. Also times one depth-2 decision.
#
# Run:  julia --project=. scripts/check_matrix_vs_directions.jl
# Pass: max relative difference < 1 % and no "FAIL" lines.
# =============================================================================
using CSV, DataFrames, Random, Statistics, Printf
include("../src/CA_STPMDP_ORS.jl")

Random.seed!(42)
mdp = StrokeMDP()
hospitals = [l for l in mdp.locations if l.type != FIELD]
pts = CSV.read("sampled_points/CA_points.csv", DataFrame)

origins = Location[]
for i in 1:15                                   # 15 pickup locations
    r = pts[rand(1:nrow(pts)), :]
    push!(origins, Location("FIELD$i", (r.Latitude, r.Longitude), -1, FIELD))
end
append!(origins, hospitals[randperm(length(hospitals))[1:5]])   # 5 hospital origins

println("Comparing Matrix vs Directions on 30 pairs ...")
rel = Float64[]; n_fail = 0; n_both_nothing = 0
for _ in 1:30
    o = origins[rand(1:length(origins))]
    h = hospitals[rand(1:length(hospitals))]
    o.latlon == h.latlon && continue
    t_dir = calculate_travel_time(o, h)              # Directions, one pair
    t_mat = cached_travel_time(mdp, o, h)            # Matrix, cached row
    if t_dir === nothing && t_mat === nothing
        global n_both_nothing += 1
        continue
    elseif t_dir === nothing || t_mat === nothing
        global n_fail += 1
        println("FAIL reachability mismatch: $(o.name) -> $(h.name)  directions=$(t_dir)  matrix=$(t_mat)")
        continue
    end
    d = abs(t_dir - t_mat) / t_dir
    push!(rel, d)
    d > 0.01 && (global n_fail += 1; println("FAIL >1 %: $(o.name) -> $(h.name)  directions=$(round(t_dir,digits=2))  matrix=$(round(t_mat,digits=2))"))
end
@printf("  pairs compared: %d   both unreachable: %d\n", length(rel), n_both_nothing)
isempty(rel) || @printf("  rel. difference: median %.4f %%   max %.4f %%\n", 100median(rel), 100maximum(rel))
println(n_fail == 0 ? "  PASS" : "  $n_fail FAIL(s)")

println("\nTiming one depth-2 decision on a fresh patient ...")
r = pts[rand(1:nrow(pts)), :]
s = PatientState(Location("FIELDt", (r.Latitude, r.Longitude), -1, FIELD), 120.0, UNKNOWN, sample_stroke_type(mdp))
t = @elapsed a = best_action(mdp, s, 2)
@printf("  depth 2 -> %s  (%.2f s, cold cache)\n", a, t)
t = @elapsed a = best_action(mdp, s, 2)
@printf("  depth 2 -> %s  (%.2f s, warm cache)\n", a, t)
