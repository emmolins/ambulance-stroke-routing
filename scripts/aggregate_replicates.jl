# =============================================================================
# aggregate_replicates.jl
#
# Pool the per-replicate CSVs into per-noise-level pooled files that the
# `CA_simulations_stats.jl` and `sensitivity_summary.jl` scripts can consume.
#
# This script auto-detects every noise level present in `simulation_results/`
# and produces one pooled CSV per noise level:
#
#   CA_simulation_results_rep1.csv ..rep10.csv         → CA_simulation_results.csv
#   CA_simulation_results_rep1_noise10.csv ..rep10_…   → CA_simulation_results_noise10.csv
#   CA_simulation_results_rep1_noise20.csv ..rep10_…   → CA_simulation_results_noise20.csv
#
# After running this once, you can:
#   julia --project=. src/CA_simulations_stats.jl              # baseline analysis
#   julia --project=. scripts/sensitivity_summary.jl           # noise comparison
#
# The multi-replicate aggregator at the bottom of the stats script still uses
# the original per-replicate CSVs to compute the IEEE-canonical between-replicate
# 95% CI — that does NOT change.
# =============================================================================

using CSV
using DataFrames

const DIR = "simulation_results"
# Region prefix of the replicate files: CA (default) or RI.
#   REGION=RI julia --project=. scripts/aggregate_replicates.jl
const REGION = uppercase(get(ENV, "REGION", "CA"))
REGION in ("CA", "RI") || error("REGION must be CA or RI")

# Matches CA_simulation_results_rep<N>(_noiseXX)?(_biasXXX)?.csv
# Group 1 = rep index, group 2 = noise tag (or nothing), group 3 = bias tag (or nothing).
const REP_PATTERN = Regex("^$(REGION)_simulation_results_rep(\\d+)(_noise\\d+)?(_bias\\d+)?\\.csv\$")

# Group replicate files by their combined sensitivity tag.
# Tag "" = deterministic baseline; "_noise10" = noise only; "_bias110" = bias only;
# "_noise10_bias110" = combined.
files_by_tag = Dict{String, Vector{Tuple{Int, String}}}()
for f in readdir(DIR)
    m = match(REP_PATTERN, f)
    m === nothing && continue
    rep_id    = parse(Int, m.captures[1])
    noise_tag = m.captures[2] === nothing ? "" : m.captures[2]
    bias_tag  = m.captures[3] === nothing ? "" : m.captures[3]
    combined  = noise_tag * bias_tag
    push!(get!(files_by_tag, combined, Tuple{Int, String}[]), (rep_id, f))
end

isempty(files_by_tag) &&
    error("No $(REGION)_simulation_results_rep*.csv files found in $DIR.")

println("Found $(length(files_by_tag)) sensitivity setting(s):")
for (tag, lst) in sort(collect(files_by_tag); by = x -> x[1])
    label = isempty(tag) ? "deterministic baseline" : "tag $tag"
    println("  $label  →  $(length(lst)) replicate(s)")
end
println()

# Aggregate each sensitivity setting independently.
for (tag, lst) in sort(collect(files_by_tag); by = x -> x[1])
    sort!(lst; by = x -> x[1])

    pooled_csv = joinpath(DIR, "$(REGION)_simulation_results$(tag).csv")
    archive    = joinpath(DIR, "$(REGION)_simulation_results$(tag).previous_archive.csv")
    label      = isempty(tag) ? "deterministic baseline" : "sensitivity tag $tag"

    println("─"^70)
    println("Aggregating $label ($(length(lst)) replicates)")
    println("─"^70)

    # Archive existing pooled file (e.g. legacy smoke test) before overwriting.
    if isfile(pooled_csv)
        n_existing = countlines(pooled_csv) - 1
        println("  Existing $(basename(pooled_csv)) has $n_existing data rows.")
        println("  Archiving to $(basename(archive)).")
        cp(pooled_csv, archive; force = true)
    end

    pooled = DataFrame()
    for (rep_id, f) in lst
        df = CSV.read(joinpath(DIR, f), DataFrame)
        df.replicate .= rep_id
        pooled = isempty(pooled) ? df : vcat(pooled, df; cols = :union)
        println("  $f  ($(nrow(df)) rows)")
    end

    CSV.write(pooled_csv, pooled)
    println("  ✓ Wrote $pooled_csv  ($(nrow(pooled)) rows × $(ncol(pooled)) cols)")
    println()
end

println("Next steps:")
println("  julia --project=. src/$(REGION)_simulations_stats.jl       # primary analysis on baseline")
println("  julia --project=. scripts/sensitivity_summary.jl    # robustness across noise levels")
