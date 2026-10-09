# Policy map: the MDP's first destination from the centre of every grid cell at
# fixed onset-to-pickup times, with the subtype unknown (the planner's view).
# Output: sampled_points/CA_policy_map.csv  (cell_i, cell_j, t_onset, action, tier)
#   julia --project=. scripts/policy_map.jl            (ORS on 8080)
#   ONSET_TIMES=60,150,240 julia --project=. scripts/policy_map.jl
using CSV, DataFrames, ProgressMeter
include("../src/CA_STPMDP_ORS.jl")

const ONSET_TIMES = parse.(Float64, split(get(ENV, "ONSET_TIMES", "60,150,240"), ","))
const OUT = "sampled_points/CA_policy_map.csv"

mdp = StrokeMDP()
cells = CSV.read("sampled_points/bay_area_grid_cells.csv", DataFrame)
tier_of = Dict(loc.name => string(loc.type) for loc in mdp.locations)
println("$(nrow(cells)) cells × $(length(ONSET_TIMES)) onset times")

rows = NamedTuple[]
prog = Progress(nrow(cells); desc = "policy map ")
for c in eachrow(cells)
    loc = Location("CELL_$(c.cell_i)_$(c.cell_j)", (c.lat_centre, c.lon_centre), -1, FIELD)
    for t in ONSET_TIMES
        s = PatientState(loc, t, UNKNOWN, LVO)     # subtype placeholder; planner marginalizes
        a = try best_action(mdp, s, 2) catch e; nothing end
        a === nothing && continue
        name = replace(enum_to_string(a), "ROUTE_" => "")
        push!(rows, (cell_i = c.cell_i, cell_j = c.cell_j, t_onset = t, action = name, tier = tier_of[name]))
    end
    next!(prog)
end
CSV.write(OUT, DataFrame(rows))
println("wrote $OUT ($(length(rows)) rows)")
