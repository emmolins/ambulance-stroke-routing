# Replay each simulated patient's pathway under each policy and record the
# onset-to-needle and onset-to-puncture times (best in-hospital follow-up for
# the drawn subtype). Reads simulation_results/<REGION>_simulation_results.csv,
# writes simulation_results/<REGION>_treatment_times.csv.
#   julia --project=. scripts/treatment_times.jl                (CA, ORS on 8080)
#   REGION=ri ORS_PORT=8081 julia --project=. scripts/treatment_times.jl
using CSV, DataFrames, ProgressMeter
const REGION = lowercase(get(ENV, "REGION", "ca"))
include(REGION == "ri" ? "../src/RI_STPMDP_ORS.jl" : "../src/CA_STPMDP_ORS.jl")

const IN  = "simulation_results/$(uppercase(REGION))_simulation_results.csv"
const OUT = "simulation_results/$(uppercase(REGION))_treatment_times.csv"
const POLICIES = (("MDP policy", :optimal_action), ("Nearest hospital", :nearest_hospital_action),
                  ("Nearest EVT-capable center", :heuristic_1_action), ("Nearest stroke center", :heuristic_2_action))

mdp = StrokeMDP()
d = CSV.read(IN, DataFrame)
println("$(nrow(d)) patients from $IN")
rows = NamedTuple[]
prog = Progress(nrow(d); desc = "treatment times ")
for (k, r) in enumerate(eachrow(d))
    loc = Location("FIELD$k", (r.start_lat, r.start_lon), -1, FIELD)
    θ = string_to_enum(String(r.stroke_type))
    s = PatientState(loc, r.t_onset, KNOWN, θ)
    for (name, col) in POLICIES
        a = string_to_enum(String(r[col]))
        sp = rand(transition(mdp, s, a))
        a1, sp2 = best_followup(mdp, sp)
        tn, tp = pathway_times(mdp, sp, sp2)
        push!(rows, (patient = k, replicate = r.replicate, t_onset = r.t_onset, stroke_type = String(r.stroke_type),
                     policy = name, first_tier = string(sp.loc.type), transferred = sp2.loc.latlon != sp.loc.latlon,
                     t_needle = tn, t_puncture = tp))
    end
    next!(prog)
end
CSV.write(OUT, DataFrame(rows))
println("wrote $OUT ($(length(rows)) rows)")
