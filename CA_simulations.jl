#=
File: CA_simulations.jl 
---------------------------------------------------------------
This script simulates stroke patient scenarios across California to compare 
an optimized MDP-derived triage policy with the current nearest-hospital policy. 
For each simulation, a random field location and stroke type are sampled, and 
patient outcomes are evaluated for both policies based on calculated rewards and 
travel times. The script aggregates results to analyze differences in outcome 
probabilities, travel efficiency, and hospital routing patterns between the two 
approaches. Outputs include CSVs and summary statistics to support further analysis 
and visualization.

=#

using CSV
using DataFrames
using StatsBase
using Random
using Plots
using StatsPlots
using ProgressMeter
using Distributions
using Random
SEED = 1234
Random.seed!(SEED)

# Import custom MDP and routing logic
include("STPMDP_ORS.jl")

# Number of patient simulations to run
N_SIMULATIONS = 1000

# Load valid points and shuffle once (in your main script)
location_pool = shuffle([(row.Latitude, row.Longitude) for row in CSV.File("sampled_points/CA_points.csv")])

# Function to sample a (lat, lon) without replacement
function sample_location!(pool::Vector{Tuple{Float64, Float64}})
    if isempty(pool)
        error("All sample locations have been used.")
    end
    return popfirst!(pool)
end

# Structure for storing each simulation's outcome and details
mutable struct SimulationResultGeneral
    rewards::Vector{Float64}  # 1: reward for best policy, #2: reward for nearest hospital (CA CURRENT), #3 reward for Heuristic #1, #4: reward for Heuristic #2
    travel_times::Vector{Float64}  # 1: travel_time for best policy, #2: travel_time for nearest hospital (CA CURRENT), #3 travel_time for Heuristic #1, #4: travel_time for Heuristic #2
    suggested_actions::Vector{String}
    start_state::PatientState
end

# Initialize the MDP for the stroke triage simulation
myMDP = StrokeMDP()

# Function to sample a random stroke type for N_SIMULATIONS
function runsims(myMDP, location_pool::Vector{Tuple{Float64, Float64}})
    results = SimulationResultGeneral[]
    progress = Progress(N_SIMULATIONS)
    n_attempts = 0
    sim_index = 1

    while length(results) < N_SIMULATIONS
        next!(progress)
        n_attempts += 1

        if isempty(location_pool)
            error("Ran out of unique locations before reaching $N_SIMULATIONS successful simulations.")
        end

        latlon = sample_location!(location_pool)

        sampled_start_state = PatientState(
            Location("FIELD$sim_index", latlon, -1, FIELD),
            30 + rand() * 240,
            UNKNOWN,
            sample_stroke_type(myMDP)
        )

        try
            # SMARTER POLICY
            recommended_action = best_action(myMDP, sampled_start_state, 2)
            if recommended_action === nothing
                @warn "No valid action for patient at $(sampled_start_state.loc.name) ($(sampled_start_state.loc.latlon))"
                continue
            end
            action_string_smarter = enum_to_string(recommended_action)
            next_state = rand(transition(myMDP, sampled_start_state, recommended_action))
            best_action_reward = reward(myMDP, sampled_start_state, recommended_action, next_state)

            # CURRENT CA POLICY
            nearest_hospital_action_string = current_CApolicy_action(myMDP, sampled_start_state)
            if nearest_hospital_action_string === nothing
                @warn "No valid nearest hospital action for patient at $(sampled_start_state.loc.name) ($(sampled_start_state.loc.latlon))"
                continue
            end
            nearest_hospital_action = string_to_enum(nearest_hospital_action_string)
            nh_next_state = rand(transition(myMDP, sampled_start_state, nearest_hospital_action))
            nearest_hospital_reward = reward(myMDP, sampled_start_state, nearest_hospital_action, nh_next_state)
            action_string_nh = enum_to_string(nearest_hospital_action)

            # HEURISTIC 1
            h1_action_string = heuristic_1_action(myMDP, sampled_start_state)
            if h1_action_string === nothing
                @warn "No valid heuristic 1 action for patient at $(sampled_start_state.loc.name) ($(sampled_start_state.loc.latlon))"
                continue
            end
            h1_action = string_to_enum(h1_action_string)
            h1_next_state = rand(transition(myMDP, sampled_start_state, h1_action))
            h1_reward = reward(myMDP, sampled_start_state, h1_action, h1_next_state)

            # HEURISTIC 2
            h2_action_string = heuristic_2_action(myMDP, sampled_start_state)
            if h2_action_string === nothing
                @warn "No valid heuristic 2 action for patient at $(sampled_start_state.loc.name) ($(sampled_start_state.loc.latlon))"
                continue
            end
            h2_action = string_to_enum(h2_action_string)
            h2_next_state = rand(transition(myMDP, sampled_start_state, h2_action))
            h2_reward = reward(myMDP, sampled_start_state, h2_action, h2_next_state)

            # Travel times
            travel_time_smarter = calculate_travel_time(sampled_start_state.loc, next_state.loc)
            travel_time_nh = calculate_travel_time(sampled_start_state.loc, nh_next_state.loc)
            travel_time_h1 = calculate_travel_time(sampled_start_state.loc, h1_next_state.loc)
            travel_time_h2 = calculate_travel_time(sampled_start_state.loc, h2_next_state.loc)

            rewards = [best_action_reward, nearest_hospital_reward, h1_reward, h2_reward]
            travel_times = [travel_time_smarter, travel_time_nh, travel_time_h1, travel_time_h2]
            actions = [action_string_smarter, action_string_nh, h1_action_string, h2_action_string]

            push!(results, SimulationResultGeneral(rewards, travel_times, actions, sampled_start_state))
            sim_index += 1
        catch e
            @warn "Simulation $sim_index failed with error: $e"
            continue
        end
    end

    println("✓ Completed $N_SIMULATIONS successful simulations using unique locations after $n_attempts attempts.")
    return results
end

results = runsims(myMDP, location_pool)

function results_to_dataframe(results)
    rows = []
    for r in results
        s = r.start_state
        push!(rows, (
            start_lat=s.loc.latlon[1],
            start_lon=s.loc.latlon[2],
            t_onset=s.t_onset,
            stroke_type=string(s.stroke_type),
            optimal_action=r.suggested_actions[1],
            optimal_action_reward=r.rewards[1],
            travel_time_optimal=r.travel_times[1],
            nearest_hospital_action=r.suggested_actions[2],
            nearest_hospital_reward=r.rewards[2],
            travel_time_nh=r.travel_times[2],
            heuristic_1_action=r.suggested_actions[3],
            heuristic_1_reward=r.rewards[3],
            travel_time_h1=r.travel_times[3],
            heuristic_2_action=r.suggested_actions[4],
            heuristic_2_reward=r.rewards[4],
            travel_time_h2=r.travel_times[4]
        ))

    end
    return DataFrame(rows)
end

df = results_to_dataframe(results)
CSV.write("simulation_results/CA_simulation_results.csv", df)
println("Simulation results written to simulation_results/CA_simulation_results.csv")



