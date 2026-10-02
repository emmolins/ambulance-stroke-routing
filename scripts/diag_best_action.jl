# One-off diagnostic: call best_action on a single patient at depth 1 and 2
# with NO try/catch so the real exception and stack trace are visible.
using CSV, DataFrames, Random
include("../src/CA_STPMDP_ORS.jl")
Random.seed!(9999)
mdp = StrokeMDP()
df = CSV.read("sampled_points/CA_points.csv", DataFrame)
row = df[1, :]
s = PatientState(Location("FIELD1", (row.Latitude, row.Longitude), -1, FIELD),
                 120.0, UNKNOWN, sample_stroke_type(mdp))
println("patient at ", (row.Latitude, row.Longitude), "  type=", s.stroke_type)
for d in (1, 2)
    println("\n--- depth $d ---")
    t = @elapsed a = best_action(mdp, s, d)
    println("depth $d -> ", a, "  (", round(t, digits=2), " s)")
end
