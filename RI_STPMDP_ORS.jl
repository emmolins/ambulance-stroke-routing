#=
RI_STPMDP_ORS.jl — MDP Implementation for Stroke Triage and Transport
------------------------------------------------------------------
This file contains the Julia implementation of a Markov Decision Process (MDP) 
for modeling stroke patient triage and ambulance routing in Rhode Island. It defines 
all core types (Location, PatientState, Actions) and loads hospital information 
from CSV. State transitions, reward functions, and optimal policy search (via forward 
search) are implemented to reflect realistic patient flows and clinical outcomes.

Hospital transfer times are computed dynamically using OpenRouteService (ORS) API 
requests, and results are cached for efficiency. Key utility functions support hospital 
lookups, travel time calculation, and simulation of patient scenarios.
=#

using Pkg
using CSV
using DataFrames
using DelimitedFiles
using DiscreteValueIteration
using LinearAlgebra
using POMDPTools
using POMDPs
using Parameters
using Plots
using Printf
using Statistics
using HTTP
using JSON
using Random
SEED =  1  
Random.seed!(SEED)

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

# Defines all possible actions: route to any hospital, or stay put
@enum Action begin
    ROUTE_RhodeIslandHospital
    ROUTE_HasbroChildrensHospital
    ROUTE_MiriamHospital
    ROUTE_RogerWilliamsHospital
    ROUTE_StJosephsHospital
    ROUTE_SouthCountyHospital
    ROUTE_KentCountyHospital
    ROUTE_LandmarkMedicalCenter
    ROUTE_WesterlyHospital
    ROUTE_BradleyHospital
    ROUTE_ButlerHospital
    ROUTE_EleanorSlaterHospitalCranston
    ROUTE_EleanorSlaterHospitalZambaronoUnitBurrillville
    ROUTE_NewportHospital
    ROUTE_OurLadyOfFatimaHospital
    ROUTE_ProvidenceVAHospital
    ROUTE_RehabilitationHospitalOfRhodeIsland
    ROUTE_WomenAndInfantsHospitalOfRhodeIsland
    STAY
end

###### =========================================================================
######  UTILITY FUNCTIONS
###### =========================================================================

# Converts a string into its Action representation (or any other enum)
# i.e. converts "ROUTE_STANFORD" to ROUTE_STANFORD
string_to_enum(str) = eval(Meta.parse(str))

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

# Returns the nearest CSC hospital to cur_loc (by car travel time)
function find_nearest_CSC(m::StrokeMDP, cur_loc::Location)
    CSCs = [loc for loc in m.locations if loc.type == CSC]
    reachable_times = Dict{Float64,Location}()

    for CSC in CSCs
        dist = nothing
        # Use cached time if available, otherwise compute and cache it
        if cur_loc.type != FIELD && haskey(m.transfer_times_dict, cur_loc.name)
            if haskey(m.transfer_times_dict[cur_loc.name], CSC.name)
                dist = m.transfer_times_dict[cur_loc.name][CSC.name]
            else
                dist = calculate_travel_time(cur_loc, CSC)
                m.transfer_times_dict[cur_loc.name][CSC.name] = dist
            end
        else
            # Initialize cache if needed and compute time
            dist = calculate_travel_time(cur_loc, CSC)
            m.transfer_times_dict[cur_loc.name] = Dict()
            m.transfer_times_dict[cur_loc.name][CSC.name] = dist
        end

        # Only keep reachable hospitals (dist !== nothing)
        if dist !== nothing
            reachable_times[dist] = CSC
        end
    end

    if isempty(reachable_times)
        return nothing
    end

    min_time = minimum(keys(reachable_times))
    return reachable_times[min_time]
end

# Returns the nearest PSC hospital to cur_loc (by car travel time)
function find_nearest_PSC(m::StrokeMDP, cur_loc::Location)
    PSCs = [loc for loc in m.locations if loc.type == PSC]
    reachable_times = Dict{Float64,Location}()

    for PSC in PSCs
        dist = nothing
        if cur_loc.type != FIELD && haskey(m.transfer_times_dict, cur_loc.name)
            if haskey(m.transfer_times_dict[cur_loc.name], PSC.name)
                dist = m.transfer_times_dict[cur_loc.name][PSC.name]
            else
                dist = calculate_travel_time(cur_loc, PSC)
                m.transfer_times_dict[cur_loc.name][PSC.name] = dist
            end
        else
            dist = calculate_travel_time(cur_loc, PSC)
            m.transfer_times_dict[cur_loc.name] = Dict()
            m.transfer_times_dict[cur_loc.name][PSC.name] = dist
        end

        # Only keep reachable hospitals
        if dist !== nothing
            reachable_times[dist] = PSC
        end
    end

    if isempty(reachable_times)
        return nothing
    end

    min_time = minimum(keys(reachable_times))
    return reachable_times[min_time]
end

# Returns the nearest clinic to cur_loc by car travel time
function find_nearest_clinic(m::StrokeMDP, cur_loc::Location)
    clinics = [loc for loc in m.locations if loc.type == CLINIC]
    reachable_times = Dict{Float64,Location}()

    for clinic in clinics
        dist = nothing
        if cur_loc.type != FIELD && haskey(m.transfer_times_dict, cur_loc.name)
            if haskey(m.transfer_times_dict[cur_loc.name], clinic.name)
                dist = m.transfer_times_dict[cur_loc.name][clinic.name]
            else
                dist = calculate_travel_time(cur_loc, clinic)
                m.transfer_times_dict[cur_loc.name][clinic.name] = dist
            end
        else
            dist = calculate_travel_time(cur_loc, clinic)
            m.transfer_times_dict[cur_loc.name] = Dict()
            m.transfer_times_dict[cur_loc.name][clinic.name] = dist
        end

        # Only include reachable clinics
        if dist !== nothing
            reachable_times[dist] = clinic
        end
    end

    if isempty(reachable_times)
        return nothing
    end

    min_time = minimum(keys(reachable_times))
    return reachable_times[min_time]
end

# Returns the nearest PSC or CSC to cur_loc by car travel time
function find_nearest_PSC_or_CSC(m::StrokeMDP, cur_loc::Location)
    potentials = [loc for loc in m.locations if loc.type == CSC || loc.type == PSC]
    reachable_times = Dict{Float64,Location}()

    for potential in potentials
        dist = nothing
        if cur_loc.type != FIELD && haskey(m.transfer_times_dict, cur_loc.name)
            if haskey(m.transfer_times_dict[cur_loc.name], potential.name)
                dist = m.transfer_times_dict[cur_loc.name][potential.name]
            else
                dist = calculate_travel_time(cur_loc, potential)
                m.transfer_times_dict[cur_loc.name][potential.name] = dist
            end
        else
            dist = calculate_travel_time(cur_loc, potential)
            m.transfer_times_dict[cur_loc.name] = Dict()
            m.transfer_times_dict[cur_loc.name][potential.name] = dist
        end

        # Only add reachable PSCs/CSCs
        if dist !== nothing
            reachable_times[dist] = potential
        end
    end

    if isempty(reachable_times)
        return nothing
    end

    min_time = minimum(keys(reachable_times))
    return reachable_times[min_time]
end

# Returns the nearest hospital (CSC, PSC, or Clinic) to cur_loc by car travel time
function find_nearest_hospital(m::StrokeMDP, cur_loc::Location)
    potentials = [loc for loc in m.locations if loc.type == CSC || loc.type == PSC || loc.type == CLINIC]
    reachable_times = Dict{Float64,Location}()

    for potential in potentials
        dist = nothing
        if cur_loc.type != FIELD && haskey(m.transfer_times_dict, cur_loc.name)
            if haskey(m.transfer_times_dict[cur_loc.name], potential.name)
                dist = m.transfer_times_dict[cur_loc.name][potential.name]
            else
                dist = calculate_travel_time(cur_loc, potential)
                m.transfer_times_dict[cur_loc.name][potential.name] = dist
            end
        else
            dist = calculate_travel_time(cur_loc, potential)
            m.transfer_times_dict[cur_loc.name] = Dict()
            m.transfer_times_dict[cur_loc.name][potential.name] = dist
        end

        # Only keep reachable hospitals
        if dist !== nothing
            reachable_times[dist] = potential
        end
    end

    if isempty(reachable_times)
        return nothing
    end

    min_time = minimum(keys(reachable_times))
    return reachable_times[min_time]
end

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
                    prob_altepase = 0.2359 + 0.0000002(t_onset_needle)^2 - 0.0004(t_onset_needle)
                else
                    prob_altepase = 0.1328
                end

            else
                # Minimum probability good outcoome for no treatment for LVO 
                prob_altepase = 0.1328
            end

            p_good_outcome = prob_altepase + ((1 - prob_altepase) * prob_EVT)

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
                prob_altepase = 0.2359 + 0.0000002(t_onset_needle)^2 - 0.0004(t_onset_needle)
            else
                prob_altepase = 0.1328
            end
        else
            prob_altepase = 0.1328
        end

        p_good_outcome_LVO = prob_altepase + ((1 - prob_altepase) * prob_EVT)


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
        p_good_outcome_hemhorragic = 0.24

        # Calculate p_good_outcome_mimic
        p_good_outcome_mimic = 0.90

        # Weighted average
        p_good_outcome = m.p_LVO * p_good_outcome_LVO + m.p_nLVO * p_good_outcome_nLVO + m.p_Hemorrhagic * p_good_outcome_hemhorragic
        +m.p_Mimic * p_good_outcome_mimic
    end
    return p_good_outcome
end

###### =========================================================================
######  POLICY SEARCH (FORWARD SEARCH & BEST ACTION)
###### =========================================================================

# Recursive forward search to estimate the maximum expected reward over a given horizon (depth).
function forward_search(m::StrokeMDP, s::PatientState, depth::Int)
    if depth == 0
        return 0.0  # Base case: no future reward
    end

    best_value = -Inf
    for a in actions(m, s)
        a = string_to_enum(a)
        sp_wrapper = transition(m, s, a)  # Get next state (deterministic)
        sp = rand(sp_wrapper)
        r = reward(m, s, a, sp)
        value = r + discount(m) * forward_search(m, sp, depth - 1)
        best_value = max(best_value, value)
    end

    return best_value
end

# Returns the action that yields the highest expected reward over the planning horizon (depth).
function best_action(m::StrokeMDP, s::PatientState, depth::Int)
    best_act = nothing
    best_value = -Inf

    for a_str in actions(m, s)
        a = string_to_enum(a_str)
        sp_wrapper = transition(m, s, a)
        sp = rand(sp_wrapper)
        r = reward(m, s, a, sp)
        value = r + discount(m) * forward_search(m, sp, depth - 1)
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

# Route patient to the nearest hospital (current CA policy).
function current_CApolicy_action(m::StrokeMDP, s::PatientState)
    cur_loc = s.loc
    nearest_hospital = find_nearest_hospital(m, cur_loc)
    if nearest_hospital === nothing
        return nothing  # No reachable hospital
    end
    return "ROUTE_" * nearest_hospital.name
end

# Route patient to the nearest Comprehensive Stroke Center (CSC).
function heuristic_1_action(m::StrokeMDP, s::PatientState)
    cur_loc = s.loc
    nearest_CSC = find_nearest_CSC(m, cur_loc)
    if nearest_CSC === nothing
        return nothing  # Can't route to any CSC
    end
    return "ROUTE_" * nearest_CSC.name
end

# Route patient to the nearest PSC or CSC.
function heuristic_2_action(m::StrokeMDP, s::PatientState)
    cur_loc = s.loc
    nearest_PSC_or_CSC = find_nearest_PSC_or_CSC(m, cur_loc)
    if nearest_PSC_or_CSC === nothing
        return nothing  # Can't route to any PSC/CSC
    end
    return "ROUTE_" * nearest_PSC_or_CSC.name
end


# Sample a random stroke type based on population statistics.
function sample_stroke_type(MDP)
    probabilities = [MDP.p_LVO, MDP.p_nLVO, MDP.p_Hemorrhagic, MDP.p_Mimic]
    stroke_types = [LVO, NLVO, HEMORRHAGIC, MIMIC]
    return rand(SparseCat(stroke_types, probabilities))
end
