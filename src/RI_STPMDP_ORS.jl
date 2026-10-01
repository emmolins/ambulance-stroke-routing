# =============================================================================
# RI_STPMDP_ORS.jl
# Markov Decision Process for stroke-patient triage and ambulance routing in the
# Rhode Island generalizability region. Structurally parallel to
# CA_STPMDP_ORS.jl; only the hospital list and patient pool differ.
#
# Clinical model: outcome probabilities follow Holodinsky JK, Williamson TS,
# Demchuk AM, et al. "Modeling stroke patient transport for all patients with
# suspected large-vessel occlusion." JAMA Neurol. 2018;75(12):1477–1486.
# doi:10.1001/jamaneurol.2018.2424 — Supplementary eAppendix, Sections B–F.
#
# Travel times: queried in real time from a locally-hosted OpenRouteService
# (ORS) instance at http://localhost:8080 (driving-car profile, OSM Rhode
# Island road network). Travel-time sensitivity (variance / systematic bias)
# is applied POST-HOC in scripts/recompute_perturbed_rewards.jl — the planner
# always sees deterministic ORS estimates.
#
# Loaded by driver scripts (RI_simulations.jl) via `include(...)`. The driver
# is responsible for seeding the RNG.
# =============================================================================

using CSV
using DataFrames
using HTTP
using JSON
using Parameters
using POMDPs
using POMDPTools     # exports `Deterministic` used in the transition function
using Random

###### =========================================================================
######  ENUMS AND DATA STRUCTURES
###### =========================================================================

@enum LocType FIELD CLINIC PSC CSC
@enum StrokeTypeKnown UNKNOWN KNOWN
@enum StrokeType LVO NLVO HEMORRHAGIC MIMIC

# File from which we will read in all the information about hospital metrics, location, etc
hospital_info_file = "hospitals/RI_hospitals.csv"

# Define Location struct
mutable struct Location
    name::String  # i.e. "STANFORD"
    latlon::Tuple{Float64, Float64}  # Location of the hospital
    performance_metric::Float64  # transfer time if CLINIC/PSC/CSC, -1 otherwise
    type::LocType  # FIELD CLINIC PSC CSC
end

# Define PatientState struct
struct PatientState
    loc::Location # Current location of patient, represented as Location struct
    t_onset::Float64 # Keeps track of time from onset to now
    stroke_type_known::StrokeTypeKnown  # UNKNOWN or KNOWN based on whether we know 
    stroke_type::StrokeType   
end

# Defines all possible actions: route to any hospital, or stay put.
# This block is kept in sync with `hospitals/RI_hospitals.csv` — the `Hospital`
# column there must match the identifier suffix here. Display-name versions
# (with spaces/punctuation) live in the CSV's `DisplayName` column.
@enum Action begin
    # Providence County
    ROUTE_RhodeIslandHospital              # Tier-1 CSC
    ROUTE_LandmarkMedicalCenter
    ROUTE_OurLadyOfFatimaHospital
    ROUTE_RogerWilliamsMedicalCenter
    ROUTE_TheMiriamHospital
    ROUTE_WomenInfantsHospital
    # Kent County
    ROUTE_KentHospital
    # Newport County
    ROUTE_NewportHospital
    # Washington County
    ROUTE_SouthCountyHospital
    ROUTE_WesterlyHospital
    STAY
end

###### =========================================================================
######  UTILITY FUNCTIONS
###### =========================================================================

# Convert an action's string label to its enum value, e.g.
# "ROUTE_StanfordHospital" -> ROUTE_StanfordHospital.
# Uses `getfield` on the module rather than `eval(Meta.parse(...))` to avoid
# code-injection if an untrusted string ever reaches this function.
string_to_enum(str) = getfield(@__MODULE__, Symbol(str))

# Converts an action into its string representation
# i.e. converts ROUTE_STANFORD to "ROUTE_STANFORD"
function enum_to_string(action)
    return (String(Symbol(action)))
end

# In: a CSV file representing hospitals
# Out: a vector of Locations
function csv_to_locations(file)
    df = CSV.read(file, DataFrame, delim=',')
    locs = []
    for row in eachrow(df)
        hospital = row["Hospital"]
        lat = row["Lat"]
        lon = row["Lon"]
        tup = (lat, lon)
        metric = Float64(row["Performance Metric"])
        type = string_to_enum(row["Type"])
        push!(locs, Location(hospital, tup, metric, type))
    end
    return locs
end

###### =========================================================================
######  MDP DEFINITION (StrokeMDP)
###### =========================================================================

# Custom MDP type
@with_kw struct StrokeMDP <: MDP{PatientState,Action}
    # Defined all constants within this StrokeMDP struct--now, we can access all fields whenever we have an instance of the MDP

    p_LVO = 0.4538  # Probability of a large vessel occlusion
    p_nLVO = 0.1092  # Probability of a non-large vessel occlusion
    p_Hemorrhagic = 0.3445  # probability of a hemorrhagic stroke
    p_Mimic = 0.0924  # Probability of a stroke mimic

    # Pull from CSV file
    locations::Vector{Location} = csv_to_locations(hospital_info_file)
    γ = 1.0  # Discount factor 

    transfer_times_dict = Dict()  # nested dictionary, key/values are like    start_loc_name : {end_loc_name : transfer time from start --> end}

end

POMDPs.discount(m::StrokeMDP) = m.γ

###### =========================================================================
######  STATE AND ACTION SPACE
###### =========================================================================

# Return all possible patient states for the MDP
# Each state is defined by location, time since onset, stroke knowledge, and stroke types
function POMDPs.states(m::StrokeMDP)
    𝒮 = Vector{PatientState}()
    for loc in m.locations
        for t_onset in 0:720
            for known in [UNKNOWN, KNOWN]
                for stroke_type in [LVO, NLVO, HEMORRHAGIC, MIMIC]
                    push!(𝒮, PatientState(loc, t_onset, known, stroke_type))
                end
            end
        end
    end
    return 𝒮
end

# Returns the set of possible actions from the given patient state
function POMDPs.actions(m::StrokeMDP, s::PatientState)
    valid_actions = String[]
    if s.loc.type == FIELD
        for hospital in m.locations
            if hospital.type != FIELD
                travel_time = calculate_travel_time(s.loc, hospital)
                if travel_time !== nothing
                    push!(valid_actions, "ROUTE_$(hospital.name)")
                end
            end
        end
    elseif s.loc.type == CLINIC
        for hospital in m.locations
            if hospital.type == PSC || hospital.type == CSC
                travel_time = calculate_travel_time(s.loc, hospital)
                if travel_time !== nothing
                    push!(valid_actions, "ROUTE_$(hospital.name)")
                end
            end
        end
        push!(valid_actions, "STAY")
    elseif s.loc.type == PSC
        for hospital in m.locations
            if hospital.type == CSC
                travel_time = calculate_travel_time(s.loc, hospital)
                if travel_time !== nothing
                    push!(valid_actions, "ROUTE_$(hospital.name)")
                end
            end
        end
        push!(valid_actions, "STAY")
    elseif s.loc.type == CSC
        return ["STAY"]
    end
    return valid_actions
end

###### =========================================================================
######  DISTANCE AND ROUTING UTILITIES
###### =========================================================================

# Compute straight-line (great-circle) distance in meters between two locations
function haversine_distance(loc1::Location, loc2::Location)
    R = 6371.0  # Earth radius in km
    lat1, lon1 = loc1.latlon
    lat2, lon2 = loc2.latlon
    lat1_rad, lon1_rad = deg2rad(lat1), deg2rad(lon1)
    lat2_rad, lon2_rad = deg2rad(lat2), deg2rad(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = sin(dlat / 2)^2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)^2
    c = 2 * atan(sqrt(a), sqrt(1 - a))
    return R * c * 1000  # distance in meters
end


# Travel-time queries are deterministic: the planner sees ORS at decision
# time. Sensitivity to travel-time variance / bias is applied POST-HOC to the
# recorded outcome (see scripts/recompute_perturbed_rewards.jl); it does not
# enter the planner's forward search.

# Returns car travel time in minutes between two locations using ORS
function calculate_travel_time(loc1::Location, loc2::Location)
    # Return nothing if distance >50 km (unroutable)
    dist_meters = haversine_distance(loc1, loc2)
    if dist_meters > 80000
        return nothing
    end

    base_url = "http://localhost:8080/ors/v2/directions/driving-car"
    start_lat, start_lon = loc1.latlon
    end_lat, end_lon = loc2.latlon
    request_url = "$base_url?&start=$start_lon,$start_lat&end=$end_lon,$end_lat"

    try
        response = HTTP.get(request_url)
        if response.status == 200
            data = JSON.parse(String(response.body))
            travel_time_seconds = data["features"][1]["properties"]["segments"][1]["duration"]
            return travel_time_seconds / 60
        else
            println("Failed to get travel time: HTTP status $(response.status)")
            return nothing
        end
    catch e
        if isa(e, HTTP.Exceptions.StatusError) && e.status == 404
            println("Warning: No routable point near coordinate $(loc1.latlon) or $(loc2.latlon)")
            return nothing
        else
            # For all other exceptions, rethrow so you see the real error
            rethrow(e)
        end
    end
end

###### =========================================================================
######  HOSPITAL LOOKUP FUNCTIONS
###### =========================================================================

"""
    cached_travel_time(m, cur_loc, target) -> Union{Float64, Nothing}

Look up — and lazily populate — the per-origin travel-time cache. Caching is
only applied for non-FIELD origins because patient pickup locations are
single-use; hospital→hospital times are reused across many decisions.
"""
function cached_travel_time(m::StrokeMDP, cur_loc::Location, target::Location)
    if cur_loc.type == FIELD
        return calculate_travel_time(cur_loc, target)
    end
    inner = get!(m.transfer_times_dict, cur_loc.name, Dict{String, Any}())
    if haskey(inner, target.name)
        return inner[target.name]
    end
    t = calculate_travel_time(cur_loc, target)
    inner[target.name] = t
    return t
end

"""
    find_nearest(m, cur_loc, pred) -> Union{Location, Nothing}

Return the nearest reachable location for which `pred(loc) == true`, by car
travel time. Returns `nothing` if no candidate is reachable.
"""
function find_nearest(m::StrokeMDP, cur_loc::Location, pred::Function)
    best_t = Inf
    best_loc = nothing
    for loc in m.locations
        pred(loc) || continue
        t = cached_travel_time(m, cur_loc, loc)
        t === nothing && continue
        if t < best_t
            best_t = t
            best_loc = loc
        end
    end
    return best_loc
end

# Type-specific shortcuts. Kept as named wrappers so call sites read clearly
# and downstream scripts that reference them by name still work.
find_nearest_CSC(m, cur_loc)        = find_nearest(m, cur_loc, l -> l.type == CSC)
find_nearest_PSC(m, cur_loc)        = find_nearest(m, cur_loc, l -> l.type == PSC)
find_nearest_clinic(m, cur_loc)     = find_nearest(m, cur_loc, l -> l.type == CLINIC)
find_nearest_PSC_or_CSC(m, cur_loc) = find_nearest(m, cur_loc, l -> l.type == PSC || l.type == CSC)
find_nearest_hospital(m, cur_loc)   = find_nearest(m, cur_loc,
                                          l -> l.type == CSC || l.type == PSC || l.type == CLINIC)

###### =========================================================================
######  TRANSITION FUNCTION
###### =========================================================================

# Returns the next patient state after taking action a from state s in the MDP.
function POMDPs.transition(m::StrokeMDP, s::PatientState, a::Action)
    cur_loc = s.loc
    if a == STAY
        dest_loc = s.loc
    else
        # Find the destination hospital by action label.
        full_term = enum_to_string(a)
        dest_loc_name = replace(full_term, "ROUTE_" => "")
        index = findfirst(loc -> loc.name == dest_loc_name, m.locations)
        if index === nothing
            return nothing
        end
        dest_loc = m.locations[index]
    end

    # Update time with travel and treatment at destination.
    treatment_time = dest_loc.performance_metric
    travel_time = calculate_travel_time(cur_loc, dest_loc)
    if travel_time === nothing
        return nothing
    end
    t_onset = s.t_onset + treatment_time + travel_time


    # Stroke type becomes known after transfer.
    known = a != STAY ? KNOWN : s.stroke_type_known

    next_state = PatientState(dest_loc, t_onset, known, s.stroke_type)
    return Deterministic(next_state)
end

###### =========================================================================
######  REWARD FUNCTION
###### =========================================================================

# Reward function: returns probability of good outcome for a patient state transition.
# CITATION: Holodinsky JK, Williamson TS, Demchuk AM, et al. Modeling Stroke Patient 
# Transport for All Patients With Suspected Large-Vessel Occlusion
function POMDPs.reward(m::StrokeMDP, s::PatientState, a::Action, sp::PatientState)

    csc_unreachable = false
    psc_unreachable = false

    # Calculate t_onset_needle and t_onset_puncture as in your original logic
    if sp.loc.type == CSC
        t_onset_needle = sp.t_onset
        t_onset_puncture = sp.t_onset
    elseif sp.loc.type == PSC
        nearest_CSC = find_nearest_CSC(m, sp.loc)
        if nearest_CSC === nothing
            csc_unreachable = true
        else
            time_to_CSC = calculate_travel_time(sp.loc, nearest_CSC)
            t_onset_puncture = sp.t_onset + time_to_CSC + nearest_CSC.performance_metric
            t_onset_needle = sp.t_onset
        end
    elseif sp.loc.type == CLINIC || sp.loc.type == FIELD
        # find nearest CSC; calculate time to CSC
        nearest_CSC = find_nearest_CSC(m, sp.loc)
        if nearest_CSC === nothing
            csc_unreachable = true
        else
            time_to_CSC = calculate_travel_time(sp.loc, nearest_CSC)
            t_onset_puncture = sp.t_onset + time_to_CSC + nearest_CSC.performance_metric
        end
        # find nearest CSC or PSC; calculate time to CSC/PSC
        nearest_PSC_or_CSC = find_nearest_PSC_or_CSC(m, sp.loc)
        if nearest_PSC_or_CSC === nothing
            psc_unreachable = true
        else
            time_to_PSC_or_CSC = calculate_travel_time(sp.loc, nearest_PSC_or_CSC)
            t_onset_needle = sp.t_onset + time_to_PSC_or_CSC + nearest_PSC_or_CSC.performance_metric
        end
    end

    if s.stroke_type_known == KNOWN
        if s.stroke_type == LVO

            # If CSC unreachable, EVT is not possible
            if csc_unreachable == false
                if t_onset_puncture < 270
                    prob_EVT = 0.3394 + 0.00000004(t_onset_puncture)^2 - 0.0002(t_onset_puncture)
                else
                    prob_EVT = 0.129
                end
            else
                prob_EVT = 0
            end

            # If PSC and CSC unreachable, alteplase is not possible
            if psc_unreachable == false || csc_unreachable == false
                if t_onset_needle < 270
                    prob_alteplase = 0.2359 + 0.0000002(t_onset_needle)^2 - 0.0004(t_onset_needle)
                else
                    prob_alteplase = 0.1328
                end

            else
                # Minimum probability good outcoome for no treatment for LVO 
                prob_alteplase = 0.1328
            end

            p_good_outcome = prob_alteplase + ((1 - prob_alteplase) * prob_EVT)

        elseif s.stroke_type == NLVO
            # If PSC and CSC unreachable, alteplase is not possible
            if psc_unreachable == false || csc_unreachable == false
                if t_onset_needle < 270
                    p_good_outcome = 0.6343 - 0.00000005(t_onset_needle)^2 - 0.0005(t_onset_needle)
                else
                    p_good_outcome = 0.4622
                end
            else 
                # Minimum probability good outcome for no treatment for nLVO
                p_good_outcome = 0.4622
            end

        elseif s.stroke_type == HEMORRHAGIC
            p_good_outcome = 0.24
        elseif s.stroke_type == MIMIC
            p_good_outcome = 0.90
        end
    else

        # If stroke type is unknown, we assume weighted probabilities

        # Calculate p_good_outcome_LVO
        if csc_unreachable == false
            if t_onset_puncture < 270
                prob_EVT = 0.3394 + 0.00000004(t_onset_puncture)^2 - 0.0002(t_onset_puncture)
            else
                prob_EVT = 0.129
            end
        else
            prob_EVT = 0
        end

        if psc_unreachable == false || csc_unreachable == false
            if t_onset_needle < 270
                prob_alteplase = 0.2359 + 0.0000002(t_onset_needle)^2 - 0.0004(t_onset_needle)
            else
                prob_alteplase = 0.1328
            end
        else
            prob_alteplase = 0.1328
        end

        p_good_outcome_LVO = prob_alteplase + ((1 - prob_alteplase) * prob_EVT)


        # Calculate p_good_outcome_nLVO
        if psc_unreachable == false || csc_unreachable == false
            if t_onset_needle < 270
                p_good_outcome_nLVO = 0.6343 - 0.00000005(t_onset_needle)^2 - 0.0005(t_onset_needle)
            else
                p_good_outcome_nLVO = 0.4622
            end
        else
            # Minimum probability good outcome for no treatment for nLVO
            p_good_outcome_nLVO = 0.4622
        end

        # Calculate p_good_outcome_hemorragic
        p_good_outcome_hemorrhagic = 0.24  # Holodinsky 2018, Section E (time-invariant)

        # Calculate p_good_outcome_mimic
        p_good_outcome_mimic = 0.90

        # Weighted average
        # Weighted average over the stroke-type prior (Holodinsky 2018, Section B).
        # All four `+` operators on the same line via trailing-operator continuation.
        p_good_outcome = m.p_LVO         * p_good_outcome_LVO +
                         m.p_nLVO        * p_good_outcome_nLVO +
                         m.p_Hemorrhagic * p_good_outcome_hemorrhagic +
                         m.p_Mimic       * p_good_outcome_mimic
    end
    return p_good_outcome
end

###### =========================================================================
######  POLICY SEARCH (FORWARD SEARCH & BEST ACTION)
###### =========================================================================

# Recursive forward search to estimate the maximum expected reward over a given horizon (depth).
#
# Information model
# -----------------
# Before any routing, the patient's stroke type is unknown to the EMS team
# (s.stroke_type_known == UNKNOWN). Any non-STAY action delivers the patient to a
# hospital, where CT/CTA imaging reveals the diagnosis — so after the transition
# sp.stroke_type_known == KNOWN.
#
# The planner must respect this information structure: at the moment of choosing
# a field action, it does not yet know which type imaging will reveal, so it
# averages the post-arrival value over the population prior on stroke types.
# After arrival (s.stroke_type_known == KNOWN), the planner conditions on the
# revealed type.
function forward_search(m::StrokeMDP, s::PatientState, depth::Int)
    if depth == 0
        return 0.0  # Base case: no future reward
    end

    types_and_probs = (
        (LVO,         m.p_LVO),
        (NLVO,        m.p_nLVO),
        (HEMORRHAGIC, m.p_Hemorrhagic),
        (MIMIC,       m.p_Mimic),
    )

    best_value = -Inf
    for a in actions(m, s)
        a = string_to_enum(a)
        sp_wrapper = transition(m, s, a)  # Get next state (deterministic)
        sp = rand(sp_wrapper)
        r = reward(m, s, a, sp)

        # UNKNOWN -> KNOWN transition = imaging reveals the type. Marginalize.
        future = if s.stroke_type_known == UNKNOWN && sp.stroke_type_known == KNOWN
            sum(
                p * forward_search(m, PatientState(sp.loc, sp.t_onset, KNOWN, t), depth - 1)
                for (t, p) in types_and_probs
            )
        else
            forward_search(m, sp, depth - 1)
        end

        value = r + discount(m) * future
        best_value = max(best_value, value)
    end

    return best_value
end

# Returns the action that yields the highest expected reward over the planning horizon (depth).
# Uses the same marginalize-on-UNKNOWN-to-KNOWN logic as `forward_search` so the action
# choice respects the prehospital information state.
function best_action(m::StrokeMDP, s::PatientState, depth::Int)
    best_act = nothing
    best_value = -Inf

    types_and_probs = (
        (LVO,         m.p_LVO),
        (NLVO,        m.p_nLVO),
        (HEMORRHAGIC, m.p_Hemorrhagic),
        (MIMIC,       m.p_Mimic),
    )

    for a_str in actions(m, s)
        a = string_to_enum(a_str)
        sp_wrapper = transition(m, s, a)
        sp = rand(sp_wrapper)
        r = reward(m, s, a, sp)

        future = if s.stroke_type_known == UNKNOWN && sp.stroke_type_known == KNOWN
            sum(
                p * forward_search(m, PatientState(sp.loc, sp.t_onset, KNOWN, t), depth - 1)
                for (t, p) in types_and_probs
            )
        else
            forward_search(m, sp, depth - 1)
        end

        value = r + discount(m) * future
        if value > best_value
            best_value = value
            best_act = a
        end
    end

    return best_act
end

###### =========================================================================
######  HEURISTIC POLICIES AND SAMPLING
###### =========================================================================

# Each heuristic policy is modeled as a *complete* EMS decision rule: it specifies
# a primary destination preference and an explicit fallback chain for when the
# primary is unreachable. This mirrors real-world EMS behavior (the crew always
# picks *some* destination) and eliminates the artificial selection bias that
# would arise from dropping patients whenever a heuristic's primary destination
# is unavailable.
#
# A policy returns `nothing` only in the catastrophic "no reachable hospital of
# any type" case — which is the same failure mode for all four policies.

# Route patient to the nearest reachable hospital of any type (current practice).
# Already complete by construction — only fails if no hospital is reachable.
function current_practice_action(m::StrokeMDP, s::PatientState)
    cur_loc = s.loc
    nearest_hospital = find_nearest_hospital(m, cur_loc)
    if nearest_hospital === nothing
        return nothing  # No reachable hospital (catastrophic — same for all policies)
    end
    return "ROUTE_" * nearest_hospital.name
end

# Heuristic 1: prefer the nearest Comprehensive Stroke Center (CSC).
# Fallback chain when no CSC is reachable: nearest PSC, then nearest hospital of
# any type.
function heuristic_1_action(m::StrokeMDP, s::PatientState)
    cur_loc = s.loc
    csc = find_nearest_CSC(m, cur_loc)
    csc !== nothing && return "ROUTE_" * csc.name
    psc = find_nearest_PSC(m, cur_loc)
    psc !== nothing && return "ROUTE_" * psc.name
    h = find_nearest_hospital(m, cur_loc)
    h !== nothing && return "ROUTE_" * h.name
    return nothing  # Catastrophic — no reachable hospital
end

# Heuristic 2: prefer the nearest Primary OR Comprehensive Stroke Center.
# Fallback when no PSC/CSC is reachable: nearest hospital of any type.
function heuristic_2_action(m::StrokeMDP, s::PatientState)
    cur_loc = s.loc
    psc_or_csc = find_nearest_PSC_or_CSC(m, cur_loc)
    psc_or_csc !== nothing && return "ROUTE_" * psc_or_csc.name
    h = find_nearest_hospital(m, cur_loc)
    h !== nothing && return "ROUTE_" * h.name
    return nothing  # Catastrophic — no reachable hospital
end


# Sample a random stroke type based on population statistics.
function sample_stroke_type(MDP)
    probabilities = [MDP.p_LVO, MDP.p_nLVO, MDP.p_Hemorrhagic, MDP.p_Mimic]
    stroke_types = [LVO, NLVO, HEMORRHAGIC, MIMIC]
    return rand(SparseCat(stroke_types, probabilities))
end
