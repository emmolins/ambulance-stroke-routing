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
using Distributions: LogNormal, cdf, quantile

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

# ---------------------------------------------------------------------------
# Onset-to-pickup time (minutes from symptom onset / last known well to the
# ambulance leaving the scene with the patient). Every cohort in the repository
# draws from this one sampler so simulations, grid, tree training and tree
# evaluation agree.
#   ONSET_DIST=lognormal  (default, primary analysis) LogNormal(median 90 min, log-SD 1.28) truncated to [30, 270].
#                          Fitted to onset-to-door for EMS-attended EVT-eligible patients
#                          (median 104, IQR 60-338 min; Hegenberg 2026) less the US median
#                          EMS transport interval of 14 min (Cash 2022 / Chari 2022).
#                          Cross-checked against LAMS-positive LA County patients (LKW to
#                          first medical contact median 26, IQR 14-64; Bosson 2023) and
#                          all-severity EMS arrivals in Cincinnati (Adeoye 2017).
#                          ONSET_MEDIAN / ONSET_LOGSD override.
#   ONSET_DIST=uniform    U(30, 270): the earlier stress-test cohort, kept for the
#                          sensitivity analysis; over-represents late presenters.
# Both samplers consume one uniform draw per patient, so seeds stay comparable.
# ---------------------------------------------------------------------------
const ONSET_DIST   = lowercase(get(ENV, "ONSET_DIST", "lognormal"))
const ONSET_MEDIAN = parse(Float64, get(ENV, "ONSET_MEDIAN", "90"))
const ONSET_LOGSD  = parse(Float64, get(ENV, "ONSET_LOGSD",  "1.28"))
const ONSET_MIN, ONSET_MAX = 30.0, 270.0
const _ONSET_CDF_LO = ONSET_DIST == "lognormal" ? cdf(LogNormal(log(ONSET_MEDIAN), ONSET_LOGSD), ONSET_MIN) : 0.0
const _ONSET_CDF_HI = ONSET_DIST == "lognormal" ? cdf(LogNormal(log(ONSET_MEDIAN), ONSET_LOGSD), ONSET_MAX) : 1.0
function sample_onset_time()
    u = rand()
    if ONSET_DIST == "lognormal"
        # inverse-CDF sampling of the truncated log-normal from a single uniform draw
        return quantile(LogNormal(log(ONSET_MEDIAN), ONSET_LOGSD), _ONSET_CDF_LO + u * (_ONSET_CDF_HI - _ONSET_CDF_LO))
    end
    return ONSET_MIN + u * (ONSET_MAX - ONSET_MIN)
end

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

# Catchment radius for the first leg from the pickup location, in km. Default: none
# (any hospital the road network reaches is a candidate first destination; long
# first legs are penalised by the outcome curves, not by a hard cutoff). Set
# FIELD_CATCHMENT_KM=80 to reproduce the earlier EMS-catchment assumption.
const FIELD_CATCHMENT_M = let v = get(ENV, "FIELD_CATCHMENT_KM", "")
    isempty(v) || lowercase(v) == "inf" ? Inf : 1000 * parse(Float64, v)
end

# Returns car travel time in minutes between two locations using ORS.
function calculate_travel_time(loc1::Location, loc2::Location)
    # Optional catchment radius for the first leg (FIELD_CATCHMENT_KM; off by
    # default). Inter-facility transfers carry no distance restriction;
    # reachability is decided by the road network (ORS returns 404 for
    # unroutable pairs).
    if loc1.type == FIELD && isfinite(FIELD_CATCHMENT_M)
        if haversine_distance(loc1, loc2) > FIELD_CATCHMENT_M
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
Unreachable destinations come back as `nothing`. The optional first-leg
catchment radius (FIELD_CATCHMENT_KM) is applied here, as in calculate_travel_time.
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
        if d === nothing || (origin.type == FIELD && isfinite(FIELD_CATCHMENT_M) && haversine_distance(origin, h) > FIELD_CATCHMENT_M)
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
#
# Sparse, terminal reward. The episode is: field --A0--> first hospital h1
# --A1--> final hospital h2 (h2 == h1 when the patient stays). No reward is paid
# on the first leg; the probability of an excellent 90-day outcome (mRS 0-1) is
# paid once, on the in-hospital decision, from the needle and puncture times
# that the completed pathway implies.
#
# Previously the field-to-hospital step was also scored with a full pathway
# outcome (assuming transfer to the nearest EVT center), and the planner summed
# that with the in-hospital step's outcome. The planner's objective was then the
# sum of two probabilities while the reported reward was one of them, so the
# policy could pick a first destination whose reported reward was below a
# comparator's. With the terminal reward the planner's objective and the
# reported value coincide: see `action_value`.
#
# Outcome curves: Holodinsky JK, Williamson TS, Demchuk AM, et al. Modeling Stroke
# Patient Transport for All Patients With Suspected Large-Vessel Occlusion.

"""
    pathway_times(m, s, sp) -> (t_needle, t_puncture)

Onset-to-needle and onset-to-puncture times implied by the completed pathway
(first hospital `s.loc` reached at `s.t_onset`, final hospital `sp.loc` reached
at `sp.t_onset`). NaN where that treatment is not available on the pathway.
"""
function pathway_times(m::StrokeMDP, s::PatientState, sp::PatientState)
    h1, h2  = s.loc, sp.loc
    stayed  = h1.latlon == h2.latlon
    t_needle   = NaN
    t_puncture = NaN
    if h1.type == PSC || h1.type == CSC
        t_needle = s.t_onset + DTN
    elseif h2.type == PSC || h2.type == CSC
        t_needle = sp.t_onset + DTN
    end
    if h2.type == CSC
        t_puncture = stayed ? s.t_onset + DTP_DIRECT : sp.t_onset + DTP_TRANSFER
    end
    return t_needle, t_puncture
end

"""
    pathway_outcome(m, s, sp) -> probability of excellent outcome

Outcome of the completed care pathway: first hospital `s.loc` reached at clock
`s.t_onset`, final hospital `sp.loc` reached at clock `sp.t_onset` (equal to `s`
when the patient stays). Conditions on `s.stroke_type` when it is known and
marginalizes over the prior otherwise.
"""
function pathway_outcome(m::StrokeMDP, s::PatientState, sp::PatientState)
    t_needle, t_puncture = pathway_times(m, s, sp)
    alteplase_possible = !isnan(t_needle)
    evt_possible       = !isnan(t_puncture)

    prob_EVT = if !evt_possible
        0.0
    elseif t_puncture < 270
        0.3394 + 0.00000004 * t_puncture^2 - 0.0002 * t_puncture
    else
        0.129
    end
    prob_alteplase = if alteplase_possible && t_needle < 270
        0.2359 + 0.0000002 * t_needle^2 - 0.0004 * t_needle
    else
        0.1328   # untreated LVO floor
    end
    p_LVO  = prob_alteplase + (1 - prob_alteplase) * prob_EVT
    p_nLVO = if alteplase_possible && t_needle < 270
        0.6343 - 0.00000005 * t_needle^2 - 0.0005 * t_needle
    else
        0.4622   # untreated nLVO floor
    end
    p_ICH   = 0.24   # Holodinsky 2018, Section E (time-invariant)
    p_mimic = 0.90

    if s.stroke_type_known == KNOWN
        s.stroke_type == LVO         && return p_LVO
        s.stroke_type == NLVO        && return p_nLVO
        s.stroke_type == HEMORRHAGIC && return p_ICH
        return p_mimic
    end
    # Stroke type unknown: weighted average over the prior (Holodinsky 2018, Section B).
    return m.p_LVO * p_LVO + m.p_nLVO * p_nLVO + m.p_Hemorrhagic * p_ICH + m.p_Mimic * p_mimic
end

# Per-step MDP reward: zero on the first leg, the pathway outcome on the
# in-hospital (stay or transfer) step.
function POMDPs.reward(m::StrokeMDP, s::PatientState, a::Action, sp::PatientState)
    s.loc.type == FIELD && return 0.0
    return pathway_outcome(m, s, sp)
end

###### =========================================================================
######  POLICY SEARCH (FORWARD SEARCH & BEST ACTION)
###### =========================================================================

# Recursive forward search (expectimax) over the two-step episode.
#
# Information model
# -----------------
# In the field the stroke type is unknown to the EMS team
# (s.stroke_type_known == UNKNOWN). Any routing action delivers the patient to a
# hospital, where imaging reveals the type, so after the transition
# sp.stroke_type_known == KNOWN. The planner respects this: when choosing the
# field action it averages the post-arrival value over the population prior,
# and after arrival it conditions on the revealed type.
#
# The episode ends after the in-hospital decision, so from a hospital state the
# search looks exactly one step ahead whatever depth is requested.
function forward_search(m::StrokeMDP, s::PatientState, depth::Int)
    depth == 0 && return 0.0
    s.loc.type != FIELD && (depth = 1)

    # Pure expected outcome of the best continuation. Tie-breaking is applied
    # only where an action is chosen (best_action, best_followup), never inside
    # the value, so a second-step tie rule cannot leak into the first-step comparison.
    best_value = -Inf
    for a_str in actions(m, s)
        a = string_to_enum(a_str)
        sp = rand(transition(m, s, a))
        best_value = max(best_value, action_value(m, s, a, sp; depth = depth))
    end
    return best_value
end

types_and_probs(m::StrokeMDP) = ((LVO, m.p_LVO), (NLVO, m.p_nLVO),
                             (HEMORRHAGIC, m.p_Hemorrhagic), (MIMIC, m.p_Mimic))

"""
    action_value(m, s, a, sp = next state; depth = 2)

Value the planner assigns to taking action `a` from state `s`: the per-step
reward plus the expected value of the best continuation. From the field with
the type unknown this is the expected probability of an excellent outcome under
the best subtype-specific follow-up, marginalized over the prior; with the type
known it is the outcome under the best follow-up for that type. This is the
quantity every policy is scored on, so the MDP policy, which maximizes it, is
never below a comparator on it. `sp` may be supplied to override the next state
(used by the travel-time perturbation analyses).
"""
function action_value(m::StrokeMDP, s::PatientState, a::Action,
                      sp::Union{PatientState, Nothing} = nothing; depth::Int = 2)
    sp === nothing && (sp = rand(transition(m, s, a)))
    r = reward(m, s, a, sp)
    future = if s.stroke_type_known == UNKNOWN && sp.stroke_type_known == KNOWN
        # Imaging reveals the type on arrival: marginalize over the prior.
        sum(p * forward_search(m, PatientState(sp.loc, sp.t_onset, KNOWN, t), depth - 1)
            for (t, p) in types_and_probs(m))
    else
        forward_search(m, sp, depth - 1)
    end
    return r + discount(m) * future
end

# Returns the action with the highest planner value over the horizon `depth`
# (ties broken toward the shorter first leg).
# Ties in expected outcome arise only when no pathway reaches treatment inside
# the outcome window (every destination scores the same floor). They are broken
# toward the more capable hospital (EVT-capable > thrombolysis-capable > other),
# then toward the shorter first leg.
const VALUE_TOL = 1e-9
tier_rank(t::LocType) = t == CSC ? 2 : t == PSC ? 1 : 0
function best_action(m::StrokeMDP, s::PatientState, depth::Int)
    best_act = nothing
    best_key = (-Inf, -1, -Inf)
    for a_str in actions(m, s)
        a = string_to_enum(a_str)
        sp = rand(transition(m, s, a))
        value = action_value(m, s, a, sp; depth = depth)
        key = (value, tier_rank(sp.loc.type), -(sp.t_onset - s.t_onset))
        if best_act === nothing || value > best_key[1] + VALUE_TOL ||
           (abs(value - best_key[1]) <= VALUE_TOL && key[2:3] > best_key[2:3])
            best_key = key
            best_act = a
        end
    end
    return best_act
end

"""
    best_followup(m, s_hosp) -> (a1, sp2)

The in-hospital decision (stay or transfer) that maximizes the terminal reward
from hospital state `s_hosp` (type known), and the resulting terminal state.
"""
function best_followup(m::StrokeMDP, s::PatientState)
    best = nothing; best_sp = nothing; best_v = -Inf
    for a_str in actions(m, s)
        a = string_to_enum(a_str)
        sp = rand(transition(m, s, a))
        v = reward(m, s, a, sp) - TRAVEL_TIEBREAK * (sp.t_onset - s.t_onset)
        v > best_v && (best_v = v; best = a; best_sp = sp)
    end
    return best, best_sp
end

"""
    tier_values(m, s) -> Dict{LocType,Float64}

Planning value (`action_value`) of the best reachable hospital in each
capability tier from field state `s`. When the top two tiers are equal the
destination tier does not affect expected outcome (no pathway reaches treatment
inside the window), and the patient is indifferent for training purposes.
"""
function tier_values(m::StrokeMDP, s::PatientState)
    best = Dict{LocType, Float64}()
    for a_str in actions(m, s)
        a = string_to_enum(a_str)
        sp = rand(transition(m, s, a))
        v = action_value(m, s, a, sp)
        best[sp.loc.type] = max(get(best, sp.loc.type, -Inf), v)
    end
    return best
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
