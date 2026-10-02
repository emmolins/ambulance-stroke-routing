# =============================================================================
# CA_STPMDP_ORS.jl
# Markov Decision Process for stroke-patient triage and ambulance routing in the
# California study region.
#
# Clinical model: outcome probabilities follow Holodinsky JK, Williamson TS,
# Demchuk AM, et al. "Modeling stroke patient transport for all patients with
# suspected large-vessel occlusion." JAMA Neurol. 2018;75(12):1477–1486.
# doi:10.1001/jamaneurol.2018.2424 — Supplementary eAppendix, Sections B–F.
#
# Travel times: queried in real time from a locally-hosted OpenRouteService
# (ORS) instance at http://localhost:8080 (driving-car profile, OSM California
# road network). Travel-time sensitivity (variance / systematic bias) is applied
# POST-HOC in scripts/recompute_perturbed_rewards.jl — the planner itself
# always sees deterministic ORS estimates, matching the dispatch-time
# information available in practice.
#
# Loaded by driver scripts (CA_simulations.jl, decision_tree_build.jl, ...) via
# `include(...)`. The driver is responsible for seeding the RNG.
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

# NSC = acute-care hospital with a 24-h ED but no stroke certification (non-stroke-center).
# PSC = thrombolysis-capable (Primary / Advanced Primary / Acute Stroke Ready).
# CSC = EVT-capable on site (Comprehensive or Thrombectomy-Capable; see hospitals/README.md).
@enum LocType FIELD NSC PSC CSC
@enum StrokeTypeKnown UNKNOWN KNOWN
@enum StrokeType LVO NLVO HEMORRHAGIC MIMIC

# File from which we will read in all the information about hospital metrics, location, etc
hospital_info_file = "hospitals/CA_hospitals.csv"

# Define Location struct
mutable struct Location
    name::String  # i.e. "STANFORD"
    latlon::Tuple{Float64,Float64}  # Location of the hospital
    performance_metric::Float64  # legacy CSV column; no longer used (see DTN/DTP/DIDO constants)
    type::LocType  # FIELD NSC PSC CSC
end

# Define PatientState struct
struct PatientState
    loc::Location # Current location of patient, represented as Location struct
    t_onset::Float64 # Keeps track of time from onset to now
    stroke_type_known::StrokeTypeKnown  # UNKNOWN or KNOWN based on whether we know 
    stroke_type::StrokeType
end

# Defines all possible actions: route to any hospital, or stay put.
# This block is kept in sync with `hospitals/CA_hospitals.csv` — the `Hospital`
# column there must match the identifier suffix here (e.g. CSV name
# "SutterEdenMedicalCenter" ↔ enum value ROUTE_SutterEdenMedicalCenter).
# Display-name versions (with spaces/punctuation) live in the CSV's
# `DisplayName` column for human-readable methods text.
@enum Action begin
    # Alameda County
    ROUTE_SutterEdenMedicalCenter
    ROUTE_AlamedaHospital
    ROUTE_AltaBatesSummitMedicalCenterAltaBatesCampus
    ROUTE_AltaBatesSummitMedicalCenterSummitCampus
    ROUTE_HighlandHospitalWilmaChanCampus
    ROUTE_KaiserFoundationHospitalFremont
    ROUTE_KaiserFoundationHospitalOaklandRichmond
    ROUTE_KaiserFoundationHospitalSanLeandro
    ROUTE_StanfordHealthCareTriValley
    ROUTE_WashingtonHospitalHealthcareSystem
    ROUTE_SanLeandroHospitalAlamedaHealthSystem
    ROUTE_StRoseHospital
    # Contra Costa County
    ROUTE_JohnMuirMedicalCenterWalnutCreekCampus
    ROUTE_JohnMuirMedicalCenterConcordCampus
    ROUTE_KaiserFoundationHospitalAntioch
    ROUTE_KaiserFoundationHospitalRichmondCampus
    ROUTE_KaiserFoundationHospitalWalnutCreek
    ROUTE_SanRamonRegionalMedicalCenter
    ROUTE_ContraCostaRegionalMedicalCenter
    ROUTE_SutterDeltaMedicalCenter
    # Marin County
    ROUTE_KaiserFoundationHospitalSanRafael
    ROUTE_MarinHealthMedicalCenter
    ROUTE_NovatoCommunityHospitalSutter
    # Napa County
    ROUTE_AdventistHealthStHelena
    ROUTE_ProvidenceQueenOfTheValleyMedicalCenter
    # San Francisco County
    ROUTE_UCSFMedicalCenter
    ROUTE_CPMCDaviesCampusSutter
    ROUTE_CPMCVanNessCampusSutter
    ROUTE_ChineseHospital
    ROUTE_KaiserFoundationHospitalSanFrancisco
    ROUTE_UCSFHealthHydeHospital
    ROUTE_UCSFHealthStanyanHospital
    ROUTE_ZuckerbergSanFranciscoGeneralHospital
    ROUTE_CPMCMissionBernalCampusSutter
    # San Mateo County
    ROUTE_KaiserFoundationHospitalRedwoodCity
    ROUTE_MillsPeninsulaMedicalCenterSutter
    ROUTE_AHMCSetonMedicalCenter
    ROUTE_KaiserFoundationHospitalSouthSanFrancisco
    ROUTE_SequoiaHospital
    ROUTE_SanMateoMedicalCenter
    # Santa Clara County
    ROUTE_ElCaminoHealthMountainView
    ROUTE_GoodSamaritanHospital
    ROUTE_KaiserFoundationHospitalSantaClara
    ROUTE_RegionalMedicalCenterOfSanJose
    ROUTE_StanfordHealthCare
    ROUTE_ElCaminoHealthLosGatos
    ROUTE_KaiserFoundationHospitalSanJose
    ROUTE_OConnorHospital
    ROUTE_SantaClaraValleyMedicalCenter
    ROUTE_StLouiseRegionalHospital
    # Solano County
    ROUTE_KaiserFoundationHospitalVacaville
    ROUTE_KaiserFoundationHospitalVallejo
    ROUTE_NorthBayMedicalCenter
    ROUTE_NorthBayVacaValleyHospital
    ROUTE_SutterSolanoMedicalCenter
    # Sonoma County
    ROUTE_HealdsburgHospitalProvidence
    ROUTE_KaiserFoundationHospitalSantaRosa
    ROUTE_PetalumaValleyHospitalProvidence
    ROUTE_ProvidenceSantaRosaMemorialHospital
    ROUTE_SonomaValleyHospital
    ROUTE_SutterSantaRosaRegionalHospital
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
const EVT_DEFINITION = lowercase(get(ENV, "EVT_DEFINITION", "certified"))
EVT_DEFINITION in ("certified", "onsite") ||
    error("EVT_DEFINITION must be 'certified' or 'onsite' (got '$EVT_DEFINITION')")

# ---------------------------------------------------------------------------
# In-hospital intervals (minutes). One value per interval for every hospital of
# a type; overridable per run through environment variables for sensitivity
# analysis. Defaults: Target: Stroke Phase III goals for DTN / DTP (45, 90 direct,
# 60 transfer-in) and the GWTG-Stroke median door-in-door-out of Royan et al.,
# Lancet Neurol 2026 (121 min, IQR 89-175). These replace the uniform 60 min
# "Performance Metric" previously applied to every arrival and transfer.
# ---------------------------------------------------------------------------
const DTN          = parse(Float64, get(ENV, "DTN_MIN",          "45"))   # door to needle
const DTP_DIRECT   = parse(Float64, get(ENV, "DTP_MIN",          "90"))   # door to puncture, direct arrival
const DTP_TRANSFER = parse(Float64, get(ENV, "DTP_TRANSFER_MIN", "60"))   # door to puncture, transferred in
const DIDO         = parse(Float64, get(ENV, "DIDO_MIN",         "121"))  # door in, door out

# Lexicographic tie-break in the planner: maximise expected outcome, and among
# destinations with (numerically) equal expected outcome prefer the shorter
# journey. 1e-7 per minute is three orders of magnitude below the smallest
# outcome difference a minute of delay produces (~2e-4), so it never
# overrides an outcome difference. Applied in the search only, not to the
# recorded reward.
const TRAVEL_TIEBREAK = parse(Float64, get(ENV, "TRAVEL_TIEBREAK", "1e-7"))

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
        # EVT_DEFINITION=onsite: treat every hospital that performs thrombectomy on
        # site (EVTOnSite == yes, including uncertified programmes) as CSC.
        # Default "certified": CSC means TJC/DNV CSC or TSC, or a LEMSA EVT designation.
        if EVT_DEFINITION == "onsite" && "EVTOnSite" in names(df) &&
           lowercase(strip(string(row["EVTOnSite"]))) == "yes"
            type = CSC
        end
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
                travel_time = cached_travel_time(m, s.loc, hospital)
                if travel_time !== nothing
                    push!(valid_actions, "ROUTE_$(hospital.name)")
                end
            end
        end
    elseif s.loc.type == NSC
        for hospital in m.locations
            if hospital.type == PSC || hospital.type == CSC
                travel_time = cached_travel_time(m, s.loc, hospital)
                if travel_time !== nothing
                    push!(valid_actions, "ROUTE_$(hospital.name)")
                end
            end
        end
        push!(valid_actions, "STAY")
    elseif s.loc.type == PSC
        for hospital in m.locations
            if hospital.type == CSC
                travel_time = cached_travel_time(m, s.loc, hospital)
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
# time, and that's all it can see. Sensitivity to travel-time variance / bias
# is applied POST-HOC to the recorded outcome (see scripts/recompute_perturbed_rewards.jl);
# it must not enter the planner's forward search.

# ORS server. ORS_PORT lets a second container (e.g. Rhode Island on 8081) be used
# without editing code.
const ORS_BASE = "http://localhost:$(get(ENV, "ORS_PORT", "8080"))"

# Returns car travel time in minutes between two locations using ORS.
function calculate_travel_time(loc1::Location, loc2::Location)
    # EMS catchment assumption: the initial transport from the pickup location
    # is restricted to facilities within 80 km. Inter-facility transfers
    # (hospital origin) carry no distance restriction; reachability is decided
    # by the road network (ORS returns 404 for unroutable pairs).
    if loc1.type == FIELD
        dist_meters = haversine_distance(loc1, loc2)
        if dist_meters > 80000
            return nothing
        end
    end

    base_url = "$(ORS_BASE)/ors/v2/directions/driving-car"
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
        elseif isa(e, HTTP.Exceptions.StatusError) && e.status == 400 &&
               occursin("2004", String(copy(e.response.body)))
            # ORS error 2004: route longer than the server's maximum_distance
            # (ors-config.yml). Treat as unroutable rather than aborting the run.
            println("Warning: ORS distance limit exceeded for $(loc1.name) -> $(loc2.name); treated as unroutable. Raise maximum_distance in ors-config.yml.")
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
    fetch_travel_row(m, origin) -> Dict{String, Union{Float64, Nothing}}

One ORS Matrix request returning car travel time (minutes) from `origin` to
every hospital in `m.locations`. Replaces one Directions request per pair
(~60 round trips and full route geometry per decision) with a single call.
Unreachable destinations come back as `nothing`. The 80 km EMS catchment
rule for pickup (FIELD) origins is applied here, as in calculate_travel_time.
Falls back to per-pair Directions calls if the Matrix endpoint fails.
"""
function fetch_travel_row(m::StrokeMDP, origin::Location)
    hospitals = [l for l in m.locations if l.type != FIELD]
    row = Dict{String, Union{Float64, Nothing}}()
    body = JSON.json(Dict(
        "locations"    => vcat([[origin.latlon[2], origin.latlon[1]]],
                               [[h.latlon[2], h.latlon[1]] for h in hospitals]),
        "sources"      => [0],
        "destinations" => collect(1:length(hospitals)),
        "metrics"      => ["duration"],
    ))
    durations = nothing
    try
        resp = HTTP.post("$(ORS_BASE)/ors/v2/matrix/driving-car",
                         ["Content-Type" => "application/json"], body)
        durations = JSON.parse(String(resp.body))["durations"][1]
    catch e
        println("Warning: ORS matrix request failed for origin $(origin.name) ($(typeof(e))); falling back to per-pair Directions calls")
        for h in hospitals
            row[h.name] = calculate_travel_time(origin, h)
        end
        return row
    end
    for (h, d) in zip(hospitals, durations)
        if d === nothing || (origin.type == FIELD && haversine_distance(origin, h) > 80000)
            row[h.name] = nothing
        else
            row[h.name] = d / 60
        end
    end
    return row
end

"""
    cached_travel_time(m, cur_loc, target) -> Union{Float64, Nothing}

Travel time from `cur_loc` to `target`, served from a per-origin cache that is
filled by one Matrix request on first use. Origins are keyed by coordinates,
so a pickup location queried dozens of times within one forward search costs
one request, and hospital-to-hospital times are reused across all decisions.
"""
function cached_travel_time(m::StrokeMDP, cur_loc::Location, target::Location)
    target.type == FIELD && return nothing
    cur_loc.latlon == target.latlon && return 0.0
    key = cur_loc.latlon
    row = get!(m.transfer_times_dict, key) do
        fetch_travel_row(m, cur_loc)
    end
    return get(row, target.name, nothing)
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
find_nearest_nsc(m, cur_loc)     = find_nearest(m, cur_loc, l -> l.type == NSC)
find_nearest_PSC_or_CSC(m, cur_loc) = find_nearest(m, cur_loc, l -> l.type == PSC || l.type == CSC)
find_nearest_hospital(m, cur_loc)   = find_nearest(m, cur_loc,
                                          l -> l.type == CSC || l.type == PSC || l.type == NSC)

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

    # Update time: a transfer out of a hospital costs the door-in-door-out
    # interval before departure; arrival itself costs nothing here, because the
    # treatment intervals (DTN / DTP) are applied in reward().
    dido = (a == STAY || cur_loc.type == FIELD) ? 0.0 : DIDO
    travel_time = cached_travel_time(m, cur_loc, dest_loc)
    if travel_time === nothing
        return nothing
    end
    t_onset = s.t_onset + dido + travel_time


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
        if s.loc.type == FIELD
            # Direct arrival at an EVT-capable centre.
            t_onset_needle   = sp.t_onset + DTN
            t_onset_puncture = sp.t_onset + DTP_DIRECT
        else
            # Transferred in. Thrombolysis was given at the sending hospital if it
            # was thrombolysis-capable; a transferred patient is pre-imaged, so the
            # shorter transfer-in door-to-puncture applies.
            t_onset_needle   = s.loc.type == NSC ? sp.t_onset + DTN : s.t_onset + DTN
            t_onset_puncture = sp.t_onset + DTP_TRANSFER
        end
    elseif sp.loc.type == PSC
        # Alteplase is given at the PSC itself, so needle time does not depend on
        # whether an onward CSC transfer exists. (Previously set only inside the
        # `else` below, leaving it undefined and crashing reward() for PSCs with
        # no CSC within the travel-time cutoff, e.g. Sonoma County.)
        t_onset_needle = sp.t_onset + DTN
        nearest_CSC = find_nearest_CSC(m, sp.loc)
        if nearest_CSC === nothing
            csc_unreachable = true
        else
            time_to_CSC = cached_travel_time(m, sp.loc, nearest_CSC)
            t_onset_puncture = sp.t_onset + DIDO + time_to_CSC + DTP_TRANSFER
        end
    elseif sp.loc.type == NSC || sp.loc.type == FIELD
        # find nearest CSC; calculate time to CSC
        nearest_CSC = find_nearest_CSC(m, sp.loc)
        if nearest_CSC === nothing
            csc_unreachable = true
        else
            time_to_CSC = cached_travel_time(m, sp.loc, nearest_CSC)
            t_onset_puncture = sp.t_onset + DIDO + time_to_CSC + DTP_TRANSFER
        end
        # No thrombolysis on site at a non-stroke-center hospital: transfer to the
        # nearest thrombolysis-capable centre for the needle as well.
        nearest_PSC_or_CSC = find_nearest_PSC_or_CSC(m, sp.loc)
        if nearest_PSC_or_CSC === nothing
            psc_unreachable = true
        else
            time_to_PSC_or_CSC = cached_travel_time(m, sp.loc, nearest_PSC_or_CSC)
            t_onset_needle = sp.t_onset + DIDO + time_to_PSC_or_CSC + DTN
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

        # Calculate p_good_outcome_hemorrhagic (Holodinsky 2018, Section E)
        p_good_outcome_hemorrhagic = 0.24

        # Calculate p_good_outcome_mimic
        p_good_outcome_mimic = 0.90

        # Weighted average over the stroke-type prior (Holodinsky 2018, Section B).
        # Note: all four `+` operators are on the SAME line via trailing-operator
        # continuation, so Julia treats this as a single expression. The earlier
        # version had a line break after `hemhorragic` without a trailing `+`,
        # which silently dropped the MIMIC term.
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

        value = r + discount(m) * future - TRAVEL_TIEBREAK * (sp.t_onset - s.t_onset)
        best_value = max(best_value, value)
    end

    return best_value
end

# Returns the action that yields the highest expected reward over the planning horizon (depth).
# Uses the same marginalize-on-UNKNOWN-to-KNOWN logic as `forward_search` so the action
# choice respects the prehospital information state (the EMS team does not have access
# to the post-arrival diagnosis when choosing the field action).
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

        value = r + discount(m) * future - TRAVEL_TIEBREAK * (sp.t_onset - s.t_onset)
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
# any type" case — which is the same failure mode for all four policies and
# represents a true model-scope limit (vanishingly rare in practice).

# Route patient to the nearest reachable hospital of any type (current CA policy).
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
# any type. Models the realistic rural-EMS pattern of escalating to whatever
# care is reachable when the preferred facility is too far.
function heuristic_1_action(m::StrokeMDP, s::PatientState)
    cur_loc = s.loc
    # Primary: nearest CSC
    csc = find_nearest_CSC(m, cur_loc)
    csc !== nothing && return "ROUTE_" * csc.name
    # Fallback 1: nearest PSC
    psc = find_nearest_PSC(m, cur_loc)
    psc !== nothing && return "ROUTE_" * psc.name
    # Fallback 2: nearest hospital of any type
    h = find_nearest_hospital(m, cur_loc)
    h !== nothing && return "ROUTE_" * h.name
    return nothing  # Catastrophic — no reachable hospital
end

# Heuristic 2: prefer the nearest Primary OR Comprehensive Stroke Center.
# Fallback when no PSC/CSC is reachable: nearest hospital of any type.
function heuristic_2_action(m::StrokeMDP, s::PatientState)
    cur_loc = s.loc
    # Primary: nearest PSC or CSC (whichever is closer)
    psc_or_csc = find_nearest_PSC_or_CSC(m, cur_loc)
    psc_or_csc !== nothing && return "ROUTE_" * psc_or_csc.name
    # Fallback: nearest hospital of any type
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
