# decision_tree_utils.jl
# ------------------------------------------------
# Helper utilities for decision tree MDP triage modeling
# (No code here will train, save, or overwrite models!)
# ------------------------------------------------

using CSV
using DataFrames
using HTTP

# --- Map action to hospital type using hospital name lookup ---
function hospital_type(a::Action)
    a_str = enum_to_string(a)
    hospital_name = replace(a_str, "ROUTE_" => "")
    csv_file_path = "hospitals/CA_hospitals"
    df = CSV.File(csv_file_path) |> DataFrame
    row_index = findfirst(df[:, :Hospital] .== hospital_name)
    if row_index !== nothing
        hospital_type = df[row_index, :Type]
    else
        error("Tried to find a hospital that did not exist!")
    end
    return eval(Meta.parse(hospital_type))
end

# --- Robust travel time calculation with retries in case of routing failures ---
function safe_calculate_travel_time(loc1::Location, loc2::Location; max_retries::Int=3)
    for attempt in 1:max_retries
        try
            return calculate_travel_time(loc1, loc2)
        catch e
            if isa(e, HTTP.Exceptions.StatusError) && e.status == 404
                println("Warning: Route not found between points (attempt $attempt/$max_retries)")
                if attempt == max_retries
                    println("Failed to find route after $max_retries attempts")
                    return nothing
                end
                sleep(0.1)
            else
                rethrow(e)
            end
        end
    end
    return nothing
end

# --- Find the nearest hospital of a given type, robust to routing errors ---
function safe_find_nearest_hospital(mdp::StrokeMDP, cur_loc::Location, hospital_type_func::Function)
    try
        hospitals = hospital_type_func(mdp, cur_loc)
        if hospitals === nothing
            return nothing, nothing
        elseif isa(hospitals, Location)
            travel_time = safe_calculate_travel_time(cur_loc, hospitals)
            if travel_time !== nothing
                return hospitals, travel_time
            else
                return nothing, nothing
            end
        else
            for hospital in hospitals
                travel_time = safe_calculate_travel_time(cur_loc, hospital)
                if travel_time !== nothing
                    return hospital, travel_time
                end
            end
            return nothing, nothing
        end
    catch e
        if isa(e, HTTP.Exceptions.StatusError) && e.status == 404
            println("Warning: Routing error in hospital finding: $(e.response.body)")
            return nothing, nothing
        else
            rethrow(e)
        end
    end
end

# --- Escape special LaTeX characters in labels (for visualization, if needed) ---
function sanitize_latex(text::String)
    text = replace(text, "<" => "\\textless{}")
    text = replace(text, ">" => "\\textgreater{}")
    text = replace(text, "&" => "\\&")
    text = replace(text, "%" => "\\%")
    text = replace(text, "#" => "\\#")
    text = replace(text, "_" => " ")
    text = replace(text, "^" => "\\textasciicircum{}")
    text = replace(text, "~" => "\\textasciitilde{}")
    text = replace(text, "\$" => "\\textdollar{}")
    return text
end

# --- Any additional utility/helper functions you want available in both train/eval can go here ---
