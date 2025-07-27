#=
File: decision_tree_build.jl
---------------------------
This script generates a dataset of synthetic stroke patient scenarios for triage modeling.
It simulates travel times, labels optimal actions using the full MDP, and engineers features for training.
A decision tree classifier is trained to approximate the optimal routing policy.
Outputs include all training data, summary statistics, and tree visualizations,
all saved to the decision_tree_output folder for further analysis and reproducibility.
=#

using CSV
using DataFrames
using DecisionTree
using AbstractTrees
using TikzGraphs
using Graphs
using TikzPictures
using Statistics
using HTTP
using JLD2
import Base.Filesystem: mkpath
using Random
SEED = 2025  
Random.seed!(SEED)

const OUTPUT_DIR = "decision_tree_output"
mkpath(OUTPUT_DIR)

include("CA_STPMDP_ORS.jl")

N_SAMPLES = 1000

# ============================================================================
# UTILITY FUNCTIONS WITH ERROR HANDLING
# ============================================================================

# Load all possible test points from CSV and set for tracking which have been used
csv_filename = "sampled_points/CA_points.csv"
df_points = CSV.File(csv_filename) |> DataFrame
all_points = [(row.Latitude, row.Longitude) for row in eachrow(df_points)]
used_points = Set{Tuple{Float64, Float64}}()

# Randomly select a location from available points
function rand_location()
    random_index = rand(1:length(all_points))
    return all_points[random_index]
end

# Map action to hospital type (CSC, PSC, or Clinic) using hospital name lookup
function hospital_type(a::Action)
    a_str = enum_to_string(a)
    hospital_name = replace(a_str, "ROUTE_" => "")
    csv_file_path = "hospitals/CA_hospitals.csv"
    df = CSV.File(csv_file_path) |> DataFrame
    row_index = findfirst(df[:, :Hospital] .== hospital_name)
    if row_index !== nothing
        hospital_type = df[row_index, :Type]
    else
        error("Tried to find a hospital that did not exist!")
    end
    return eval(Meta.parse(hospital_type))
end

# Robust travel time calculation with retries in case of routing failures
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

# Find the nearest hospital of a given type, robust to routing errors
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


# Sample a routable patient state, ensuring at least one reachable hospital of each type
function sample_routable_patient_state(mdp::StrokeMDP; max_attempts::Int=50)
    for attempt in 1:max_attempts
        sampled_s = PatientState(
            Location("FIELD1", rand_location(), -1, FIELD), 
            rand() * 270, 
            UNKNOWN, 
            sample_stroke_type(mdp)
        )
        csc_hospital, t_nearest_CSC = safe_find_nearest_hospital(mdp, sampled_s.loc, find_nearest_CSC)
        psc_hospital, t_nearest_PSC = safe_find_nearest_hospital(mdp, sampled_s.loc, find_nearest_PSC)
        clinic_hospital, t_nearest_clinic = safe_find_nearest_hospital(mdp, sampled_s.loc, find_nearest_clinic)
        if t_nearest_CSC !== nothing || t_nearest_PSC !== nothing || t_nearest_clinic !== nothing
            return sampled_s, t_nearest_CSC, t_nearest_PSC, t_nearest_clinic
        end        
        if attempt % 10 == 0
            println("Attempt $attempt/$max_attempts: Still searching for routable patient state...")
        end
    end
    println("Warning: Could not find a routable patient state after $max_attempts attempts")
    return nothing, nothing, nothing, nothing
end

# ============================================================================
# TREE VISUALIZATION FUNCTIONS (Credit: Robert Moss)
# ============================================================================

# Use '--' for edges in TikzGraphs (produces simpler diagrams)
TikzGraphs.edge_str(g) = "--"

# Custom plot function for TikzGraphs—generates a TikZ/LaTeX tree from a graph and labels
function TikzGraphs.plot(g; layout::Layouts.Layout=Layouts.Layered(), 
                        labels::Vector{T}=map(string, vertices(g)), 
                        edge_labels::Dict = Dict(), 
                        node_styles::Dict = Dict(), 
                        node_style="", 
                        edge_styles::Dict = Dict(), 
                        edge_style="", 
                        options="", 
                        graph_options="", 
                        prepend_preamble::String="") where T<:AbstractString
    
    # Sanitize node labels for LaTeX safety
    sanitized_labels = [sanitize_latex(label) for label in labels]
    
    o = IOBuffer()
    println(o, "\\graph [$(TikzGraphs.layoutname(layout)), $(TikzGraphs.options_str(layout)), $graph_options] {")
    for v in vertices(g)
        TikzGraphs.nodeHelper(o, v, sanitized_labels, node_styles, node_style)
    end
    println(o, ";")
    for e in edges(g)
        a = src(e)
        b = dst(e)
        print(o, "$a $(TikzGraphs.edge_str(g))")
        TikzGraphs.edgeHelper(o, a, b, edge_labels, edge_styles, edge_style)
        println(o, "$b;")
    end
    println(o, "};")
    mypreamble = prepend_preamble * TikzGraphs.preamble * "\n\\usegdlibrary{$(TikzGraphs.libraryname(layout))}"
    TikzGraphs.TikzPicture(String(take!(o)), preamble=mypreamble, options=options)
end

# Escape special LaTeX characters in labels
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

# Helper for computing tree size (node/leaf counts)
Base.length(tree::InfoNode) = 1 + sum(length(child) for child in children(tree))
Base.length(leaf::InfoLeaf) = 1

# Convert a leaf node to a string (optionally with rounding)
function node2str(leaf::InfoLeaf; rounding=false, sigdigits=3)
    if hasproperty(leaf, :info) && hasproperty(leaf.info, :classlabels)
        return string(leaf.info.classlabels[leaf.leaf.majority])
    else
        majority = leaf.leaf.majority
        if rounding
            return string(round(majority; sigdigits))
        else
            return string(majority)
        end
    end
end

# Convert an internal node to a string (feature threshold test)
function node2str(tree::InfoNode; rounding=false, sigdigits=3)
    val = tree.node.featval
    if rounding
        val = round(val; sigdigits)
    end
    return string(tree.info.featurenames[tree.node.featid], " ≤ ", val)
end

# Convert trained model to Tikz graph with full label information
function tree2graph(model::Union{DecisionTreeClassifier,DecisionTreeRegressor}, features, classes; rounding=false, sigdigits=3)
    tree = DecisionTree.wrap(model.root, (featurenames=features, classlabels=classes))
    g, tree_labels = tree2graph(tree; rounding, sigdigits)
    return TikzGraphs.plot(g; labels=tree_labels)
end

# Overload for just features (no class labels)
function tree2graph(model::Union{DecisionTreeClassifier,DecisionTreeRegressor}, features; rounding=false, sigdigits=3)
    tree = DecisionTree.wrap(model.root, (featurenames=features,))
    g, tree_labels = tree2graph(tree; rounding, sigdigits)
    return TikzGraphs.plot(g; labels=tree_labels)
end

# Recursively traverse InfoNode tree and build a SimpleGraph and node labels
function tree2graph(tree::InfoNode, g=SimpleGraph(length(tree)), ids=[1], labels=[]; rounding=false, sigdigits=3)
    if isempty(labels)
        labels = [node2str(tree; rounding, sigdigits)]
    end
    i_root = ids[end]
    for child in children(tree)
        push!(ids, length(ids)+1)
        push!(labels, node2str(child; rounding, sigdigits))
        add_edge!(g, i_root, ids[end])
        tree2graph(child, g, ids, labels; rounding, sigdigits)
    end
    return g, labels
end

# No further action required for leaves (end recursion)
tree2graph(tree::InfoLeaf, g=SimpleGraph(length(tree)), ids=[], labels=[]; kwargs...) = nothing

# ============================================================================
# SIMPLE TREE REPRESENTATION (FALLBACK)
# ============================================================================
# Generates a human-readable, text-based representation of a trained decision tree.
# Useful as a fallback when graphical/TikZ visualization is unavailable.

# Entry point: wraps the trained model and calls the recursive text generator
function create_simple_tree_text(model::DecisionTreeClassifier, feature_names::Vector{String}, action_labels::Vector{String})
    tree = DecisionTree.wrap(model.root, (featurenames=feature_names, classlabels=action_labels))
    return create_simple_tree_text_recursive(tree, 0)
end

# Recursively build a text tree, indenting based on depth and using UTF-8 tree symbols
function create_simple_tree_text_recursive(node, depth::Int, prefix::String="")
    indent = "  " * "│ "^depth  # Indentation and branch lines

    if isa(node, InfoLeaf)
        # Leaf node: print the majority decision/action
        if hasproperty(node, :info) && hasproperty(node.info, :classlabels)
            action = node.info.classlabels[node.leaf.majority]
        else
            action = string(node.leaf.majority)
        end
        return indent * "└─ Decision: " * action * "\n"
    else
        # Internal node: print feature and split threshold, then recurse on children
        result = ""
        val = round(node.node.featval, digits=2)
        feature_name = node.info.featurenames[node.node.featid]
        result *= indent * "├─ " * feature_name * " ≤ " * string(val) * "?\n"
        
        child_nodes = children(node)
        for (i, child) in enumerate(child_nodes)
            # Pass YES/NO as a prefix, though not printed in this structure
            result *= create_simple_tree_text_recursive(child, depth + 1, (i == 1 ? "YES: " : "NO:  "))
        end
        return result
    end
end

# ============================================================================
# MAIN EXECUTION WITH IMPROVED FEATURE ENCODING
# ============================================================================

# Initialize arrays to store feature data and labels
time_to_CSCs = Union{Float64, Missing}[]
time_to_PSCs = Union{Float64, Missing}[]
time_to_clinics = Union{Float64, Missing}[]
t_onsets = Float64[]
csc_reachable = Bool[]
psc_reachable = Bool[]
clinic_reachable = Bool[]
labels = Int[]

mdp = StrokeMDP()

println("Generating $(N_SAMPLES) samples with improved feature encoding...")

global successful_samples = 0
global failed_attempts = 0

while successful_samples < N_SAMPLES
    global successful_samples, failed_attempts, N_SAMPLES

    # Sample a routable patient state with valid hospital routes
    sampled_s, t_nearest_CSC, t_nearest_PSC, t_nearest_clinic = sample_routable_patient_state(mdp)

    if sampled_s === nothing
        failed_attempts += 1
        println("Failed to generate routable sample (attempt $(failed_attempts))")
        if failed_attempts > N_SAMPLES
            println("Too many failed attempts. Reducing target sample size.")
            N_SAMPLES = successful_samples
            break
        end
        continue
    end

    # Instead of 0.0, use `missing` for unreachable hospitals
    t_csc = t_nearest_CSC !== nothing ? t_nearest_CSC : missing
    t_psc = t_nearest_PSC !== nothing ? t_nearest_PSC : missing
    t_clinic = t_nearest_clinic !== nothing ? t_nearest_clinic : missing

    # Store patient features including reachability
    push!(time_to_CSCs, t_csc)
    push!(time_to_PSCs, t_psc)
    push!(time_to_clinics, t_clinic)
    push!(t_onsets, sampled_s.t_onset)
    
    # NEW: Store reachability as boolean features
    push!(csc_reachable, t_nearest_CSC !== nothing)
    push!(psc_reachable, t_nearest_PSC !== nothing)
    push!(clinic_reachable, t_nearest_clinic !== nothing)

    try
        # Get the optimal action label for this patient
        a = best_action(mdp, sampled_s, 2)
        type = hospital_type(a)
        label = if type == CSC
            1
        elseif type == PSC
            2
        else
            3
        end
        push!(labels, label)
        successful_samples += 1

        # Track this patient location as used
        push!(used_points, sampled_s.loc.latlon)

        # Periodic progress update
        if successful_samples % max(1, N_SAMPLES ÷ 10) == 0
            println("Successfully processed $(successful_samples)/$(N_SAMPLES) samples ($(failed_attempts) failures)")
        end

    catch e
        println("Error getting best action for sample: $e")
        # Roll back pushes if action fails
        pop!(time_to_CSCs)
        pop!(time_to_PSCs)
        pop!(time_to_clinics)
        pop!(t_onsets)
        pop!(csc_reachable)  # NEW: Clean up boolean flags too
        pop!(psc_reachable)
        pop!(clinic_reachable)
        failed_attempts += 1
        continue
    end
end

global N_SAMPLES = successful_samples

if N_SAMPLES == 0
    error("No successful samples generated. Check your data files and routing service.")
end

# Replace missing values with a large number before building features
time_to_CSCs_clean = [ismissing(x) ? 1e8 : x for x in time_to_CSCs]
time_to_PSCs_clean = [ismissing(x) ? 1e8 : x for x in time_to_PSCs]
time_to_clinics_clean = [ismissing(x) ? 1e8 : x for x in time_to_clinics]


println("Successfully generated $(N_SAMPLES) samples with $(failed_attempts) failed attempts")

# ============================================================================
# EXPORT TRAIN/TEST POINT SPLITS
# ============================================================================
# Split all possible locations into a training set (used in sampling) and
# a test set (unused locations), then save each as CSV for downstream analysis.

# Training set: points used in sampling
train_points = [p for p in all_points if p in used_points]
train_df = DataFrame(Latitude = [p[1] for p in train_points],
                     Longitude = [p[2] for p in train_points])
CSV.write(joinpath(OUTPUT_DIR, "decision_tree_train.csv"), train_df)
println("✓ Saved decision_tree_train.csv with $(nrow(train_df)) points")

# Test set: points not used in sampling
test_points = [p for p in all_points if !(p in used_points)]
test_df = DataFrame(Latitude = [p[1] for p in test_points],
                    Longitude = [p[2] for p in test_points])
CSV.write(joinpath(OUTPUT_DIR, "decision_tree_test.csv"), test_df)
println("✓ Saved decision_tree_test.csv with $(nrow(test_df)) points")

# ============================================================================
# IMPROVED FEATURE ENGINEERING
# ============================================================================

println("Engineering features with boolean reachability...")

# Only compute differences and ratios when both hospitals are reachable
diffs_CSC_PSC = Float64[]
diffs_CSC_Clinic = Float64[]
diffs_PSC_Clinic = Float64[]
ratios_CSC_PSC = Float64[]
ratios_CSC_Clinic = Float64[]
ratios_PSC_Clinic = Float64[]

for i in 1:N_SAMPLES
    # Differences: only meaningful when both hospitals are reachable
    push!(diffs_CSC_PSC, 
          (csc_reachable[i] && psc_reachable[i]) ? abs(time_to_CSCs_clean[i] - time_to_PSCs_clean[i]) : 0.0)
    push!(diffs_CSC_Clinic, 
          (csc_reachable[i] && clinic_reachable[i]) ? abs(time_to_CSCs_clean[i] - time_to_clinics_clean[i]) : 0.0)
    push!(diffs_PSC_Clinic, 
          (psc_reachable[i] && clinic_reachable[i]) ? abs(time_to_PSCs_clean[i] - time_to_clinics_clean[i]) : 0.0)
    
    # Ratios: only meaningful when denominator hospital is reachable
    push!(ratios_CSC_PSC, 
          (csc_reachable[i] && psc_reachable[i] && time_to_PSCs_clean[i] > 0) ? time_to_CSCs_clean[i] / time_to_PSCs_clean[i] : 1.0)
    push!(ratios_CSC_Clinic, 
          (csc_reachable[i] && clinic_reachable[i] && time_to_clinics_clean[i] > 0) ? time_to_CSCs_clean[i] / time_to_clinics_clean[i] : 1.0)
    push!(ratios_PSC_Clinic, 
          (psc_reachable[i] && clinic_reachable[i] && time_to_clinics_clean[i] > 0) ? time_to_PSCs_clean[i] / time_to_clinics_clean[i] : 1.0)
end

# OPTION 1: Include reachability as explicit boolean features
USE_EXTENDED_FEATURES = true
INCLUDE_REACHABILITY_FLAGS = true  # NEW: Toggle for boolean features

if USE_EXTENDED_FEATURES && INCLUDE_REACHABILITY_FLAGS
    println("Using extended feature set with reachability flags...")
    features = hcat(
        time_to_CSCs_clean, time_to_PSCs_clean, time_to_clinics_clean, t_onsets,
        Float64.(csc_reachable), Float64.(psc_reachable), Float64.(clinic_reachable),  # Convert Bool to Float64
        diffs_CSC_PSC, diffs_CSC_Clinic, diffs_PSC_Clinic, 
        ratios_CSC_PSC, ratios_CSC_Clinic, ratios_PSC_Clinic
    )
    feature_names = [
        "Time to CSC", "Time to PSC", "Time to Clinic", "Time since onset",
        "CSC Reachable", "PSC Reachable", "Clinic Reachable",  # NEW: Clear boolean features
        "Diff CSC-PSC", "Diff CSC-Clinic", "Diff PSC-Clinic",
        "Ratio CSC/PSC", "Ratio CSC/Clinic", "Ratio PSC/Clinic"
    ]
elseif USE_EXTENDED_FEATURES
    println("Using extended feature set without explicit reachability flags...")
    features = hcat(
        time_to_CSCs_clean, time_to_PSCs_clean, time_to_clinics_clean, t_onsets, 
        diffs_CSC_PSC, diffs_CSC_Clinic, diffs_PSC_Clinic, 
        ratios_CSC_PSC, ratios_CSC_Clinic, ratios_PSC_Clinic
    )
    feature_names = [
        "Time to CSC", "Time to PSC", "Time to Clinic", "Time since onset",
        "Diff CSC-PSC", "Diff CSC-Clinic", "Diff PSC-Clinic",
        "Ratio CSC/PSC", "Ratio CSC/Clinic", "Ratio PSC/Clinic"
    ]
else
    println("Using basic feature set with reachability flags...")
    features = hcat(
        time_to_CSCs_clean, time_to_PSCs_clean, time_to_clinics_clean, t_onsets,
        Float64.(csc_reachable), Float64.(psc_reachable), Float64.(clinic_reachable)
    )
    feature_names = [
        "Time to CSC", "Time to PSC", "Time to Clinic", "Time since onset",
        "CSC Reachable", "PSC Reachable", "Clinic Reachable"
    ]
end

# ============================================================================
# MODEL TRAINING AND EVALUATION
# ============================================================================
# Train the decision tree classifier on the generated features and labels.
# Print tree structure, feature mappings, action label definitions, and
# basic statistics for each feature.

println("Training decision tree...")

# Train decision tree classifier with specified depth
model = DecisionTreeClassifier(max_depth=4)
DecisionTree.fit!(model, features, labels)

# Print tree structure to the console for inspection
println("\nDecision Tree Structure:")
DecisionTree.print_tree(model)

# Print feature index-to-name mapping for transparency
println("\n" * "="^60)
println("FEATURE MAPPING:")
println("="^60)
for (i, name) in enumerate(feature_names)
    println("Feature $i: $name")
end
println("="^60)

# Define and print action labels (output classes)
action_labels = ["Route_CSC", "Route_PSC", "Route_Clinic"]
println("\nACTION LABELS:")
for (i, action) in enumerate(action_labels)
    println("Label $i: $action")
end
println("="^60)

# Print descriptive statistics for each feature column
println("\nFEATURE STATISTICS:")
println("="^60)
for (i, name) in enumerate(feature_names)
    feature_data = features[:, i]
    println("$name:")
    println("  Min: $(round(minimum(feature_data), digits=2))")
    println("  Max: $(round(maximum(feature_data), digits=2))")
    println("  Mean: $(round(mean(feature_data), digits=2))")
    println("  Std: $(round(std(feature_data), digits=2))")
    println()
end
println("="^60)

# ============================================================================
# VISUALIZATION AND OUTPUT
# ============================================================================
# Generate and save visualizations of the trained decision tree. 
# Both a LaTeX/TikZ diagram and a simple text-based version are created.
# Falls back to text representation if TikZ fails.

println("Generating tree visualization...")

try
    # Generate TikZ/graph representation of the tree
    tikz = tree2graph(model, feature_names, action_labels)
    
    # Save LaTeX/TikZ source to file for manual compilation
    TikzGraphs.TikzPictures.save(TikzGraphs.TEX(joinpath(OUTPUT_DIR, "decision_tree")), tikz)
    println("Tree saved as decision_tree.tex")
    
    # Also create and save a simple text-based version for readability
    println("\nSimple tree representation:")
    simple_tree_text = create_simple_tree_text(model, feature_names, action_labels)
    open(joinpath(OUTPUT_DIR, "decision_tree_simple.txt"), "w") do file
        write(file, simple_tree_text)
    end    
    println("Simple tree saved as decision_tree_simple.txt")
    
    # Try to save a compiled PDF (requires LaTeX installation)
    try
        TikzGraphs.TikzPictures.save(TikzGraphs.PDF(joinpath(OUTPUT_DIR, "decision_tree")), tikz)
        println("Tree saved as decision_tree.pdf")
    catch e
        println("Could not save PDF - this is normal if LaTeX is not installed")
        println("You can compile the .tex file manually with a LaTeX installation")
    end    
    
catch e
    println("Error generating TikZ visualization: ", e)
    println("Creating fallback simple tree representation...")
    
    # If TikZ visualization fails, just save the text-based version
    try
        simple_tree_text = create_simple_tree_text(model, feature_names, action_labels)
        println("\nSimple tree representation:")
        println(simple_tree_text)
        open(joinpath(OUTPUT_DIR, "decision_tree_simple.txt"), "w") do file
            write(file, simple_tree_text)
        end
        println("Simple tree saved as decision_tree_simple.txt")
    catch e2
        println("Error creating simple tree: ", e2)
    end
end

# ============================================================================
# UPDATED DATA EXPORT
# ============================================================================

println("Saving comprehensive training data with reachability info...")

# Define action labels first (moved from later in original code)
action_labels = ["Route_CSC", "Route_PSC", "Route_Clinic"]

# Create enhanced training DataFrame
sample_ids = collect(1:N_SAMPLES)

training_df = DataFrame(
    Sample_ID = sample_ids,
    Time_to_CSC_min = round.(time_to_CSCs_clean, digits=2),
    Time_to_PSC_min = round.(time_to_PSCs_clean, digits=2),
    Time_to_Clinic_min = round.(time_to_clinics_clean, digits=2),
    Time_since_onset_min = round.(t_onsets, digits=2),
    CSC_Reachable = csc_reachable,  # NEW: Include reachability info in export
    PSC_Reachable = psc_reachable,
    Clinic_Reachable = clinic_reachable,
    Predicted_Label = labels,
    Predicted_Action = [action_labels[l] for l in labels]
)

# Add extended features if used
if USE_EXTENDED_FEATURES
    training_df.Diff_CSC_PSC = round.(diffs_CSC_PSC, digits=2)
    training_df.Diff_CSC_Clinic = round.(diffs_CSC_Clinic, digits=2)
    training_df.Diff_PSC_Clinic = round.(diffs_PSC_Clinic, digits=2)
    training_df.Ratio_CSC_PSC = round.(ratios_CSC_PSC, digits=3)
    training_df.Ratio_CSC_Clinic = round.(ratios_CSC_Clinic, digits=3)
    training_df.Ratio_PSC_Clinic = round.(ratios_PSC_Clinic, digits=3)
end

# Enhanced analysis: identify truly fastest reachable hospital
training_df.Fastest_Reachable_Hospital = map(1:N_SAMPLES) do i
    reachable_times = []
    reachable_hospitals = []
    
    if csc_reachable[i]
        push!(reachable_times, time_to_CSCs_clean[i])
        push!(reachable_hospitals, "CSC")
    end
    if psc_reachable[i]
        push!(reachable_times, time_to_PSCs_clean[i])
        push!(reachable_hospitals, "PSC")
    end
    if clinic_reachable[i]
        push!(reachable_times, time_to_clinics_clean[i])
        push!(reachable_hospitals, "Clinic")
    end
    
    if !isempty(reachable_times)
        return reachable_hospitals[argmin(reachable_times)]
    else
        return "None"
    end
end

training_df.Decision_Matches_Fastest_Reachable = 
    training_df.Predicted_Action .== ("Route_" .* training_df.Fastest_Reachable_Hospital)


CSV.write(joinpath(OUTPUT_DIR, "training_data_detailed.csv"), training_df)
println("✓ Enhanced training data with reachability saved as training_data_detailed.csv")

# Remove missing values from time arrays for summary stats
clean_time_to_CSCs = [t for (t, r) in zip(time_to_CSCs_clean, csc_reachable) if r]
clean_time_to_PSCs = [t for (t, r) in zip(time_to_PSCs_clean, psc_reachable) if r]
clean_time_to_clinics = [t for (t, r) in zip(time_to_clinics_clean, clinic_reachable) if r]

# Export summary stats for quick sanity checks
summary_stats = DataFrame(
    Metric = ["Total Samples", "Failed Attempts", "Route to CSC", "Route to PSC", "Route to Clinic", 
              "Avg Time to CSC", "Avg Time to PSC", "Avg Time to Clinic", 
              "Avg Onset Time", "Decision Matches Fastest", "Decision Different from Fastest"],
    Value = [
        N_SAMPLES,
        failed_attempts,
        sum(training_df.Predicted_Action .== "Route_CSC"),
        sum(training_df.Predicted_Action .== "Route_PSC"), 
        sum(training_df.Predicted_Action .== "Route_Clinic"),
        round(mean(clean_time_to_CSCs), digits=2),
        round(mean(clean_time_to_PSCs), digits=2),
        round(mean(clean_time_to_clinics), digits=2),
        round(mean(t_onsets), digits=2),
        sum(training_df.Decision_Matches_Fastest_Reachable),
        sum(.!training_df.Decision_Matches_Fastest_Reachable)
    ]
)

CSV.write(joinpath(OUTPUT_DIR, "training_summary.csv"), summary_stats)
println("✓ Summary statistics saved as training_summary.csv")

# Save decision breakdown by onset category and predicted action

decision_breakdown = combine(groupby(training_df, :Predicted_Action), nrow => :Count)
CSV.write(joinpath(OUTPUT_DIR,"decision_breakdown_by_action.csv"), decision_breakdown)
println("✓ Decision breakdown by predicted action saved as decision_breakdown_by_action.csv")

# Print final export summary
println("\n" * "="^60)
println("EXCEL ANALYSIS FILES CREATED:")
println("="^60)
println("1. training_data_detailed.csv     - Main dataset with all features")
println("2. training_summary.csv           - Quick statistics overview") 
println("3. decision_breakdown_by_action.csv - Decisions by predicted action")
println("="^60)
println("ROBUSTNESS SUMMARY:")
println("• Successfully generated $(N_SAMPLES) valid samples")
println("• Handled $(failed_attempts) routing failures gracefully")
println("• All samples have valid route to at least one hospital; unreachable options are encoded as very large travel times.")
println("="^60)

println("\nAnalysis complete!")
println("- Decision tree trained with $(N_SAMPLES) valid samples")
println("- $(failed_attempts) routing failures handled gracefully")
println("- Tree visualization saved (if successful)")
println("- Training data exported for manual inspection")

# Save the trained model and metadata for later use
@save joinpath(OUTPUT_DIR, "decision_tree_model.jld2") model feature_names action_labels
