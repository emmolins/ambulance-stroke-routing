# setup.jl
#
# One-shot environment setup for the ambulance-stroke-routing project.
#
# Run this once after cloning, from the repo root:
#     julia --project=. setup.jl
#
# It activates the local project environment, adds every package the source
# files import, and instantiates a Manifest.toml so dependency versions are
# pinned for publication-grade reproducibility.

using Pkg

# Activate the project rooted at this file's directory.
Pkg.activate(@__DIR__)

# Packages imported across the .jl files in this repo.
# (Pkg, Printf, Random, Statistics, LinearAlgebra, DelimitedFiles are stdlib
#  in modern Julia and do not need to be added — they are available with
#  `using` once Julia is installed.)
deps = [
    "AbstractTrees",
    "CSV",
    "DataFrames",
    "DecisionTree",
    "Distributions",
    "GMT",
    "Graphs",
    "HTTP",
    "JLD2",
    "JSON",
    "Measures",
    "POMDPs",
    "POMDPTools",
    "Parameters",
    "Plots",
    "ProgressBars",
    "ProgressMeter",
    "StatsBase",
    "TikzGraphs",
    "TikzPictures",
]

println("Adding $(length(deps)) packages to the project environment...")
Pkg.add(deps)

println("\nPrecompiling...")
Pkg.precompile()

println("\nDone. Project.toml and Manifest.toml have been written/updated.")
println("From now on, start Julia with:  julia --project=.")
