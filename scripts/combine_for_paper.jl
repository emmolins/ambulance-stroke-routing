# =============================================================================
# combine_for_paper.jl
#
# Join the marginalized and realized analysis CSVs into manuscript-ready
# combined tables, with both reward versions side by side for direct paper
# inclusion.
#
# Input:  simulation_results/CA_paired_comparisons*.csv (marginalized)
#         simulation_results/CA_paired_comparisons*_realized.csv (realized)
# Output: simulation_results/PAPER_TABLE_*.csv
#
# Run:    julia --project=. combine_for_paper.jl
# =============================================================================

using CSV
using DataFrames

const DIR = "simulation_results"

# --- helpers ----------------------------------------------------------------

"""
Append `_marg` and `_real` suffixes to non-key columns of two DataFrames,
then horizontally join them by the key columns.
"""
function combine_pair(marg::DataFrame, real::DataFrame; keys::Vector{Symbol})
    # Rename non-key columns with suffixes
    function tag!(df::DataFrame, suffix::String)
        non_keys = setdiff(Symbol.(names(df)), keys)
        rename!(df, Dict(c => Symbol(string(c) * suffix) for c in non_keys))
    end
    marg_tagged = copy(marg); tag!(marg_tagged, "_marg")
    real_tagged = copy(real); tag!(real_tagged, "_real")
    # Inner join on the key columns; both versions are guaranteed aligned by analysis.
    combined = innerjoin(marg_tagged, real_tagged, on = keys)
    return combined
end

"""
Round selected numeric columns to a fixed number of digits in-place.
Convenience for manuscript-table readability.
"""
function round_cols!(df::DataFrame, digits::Int = 5)
    for c in names(df)
        if eltype(df[!, c]) <: AbstractFloat
            df[!, c] = round.(df[!, c]; digits = digits)
        end
    end
    return df
end

"""
Read a CSV if it exists; return nothing otherwise (with a warning).
"""
function try_read(path::String)
    if !isfile(path)
        @warn "Missing $path"
        return nothing
    end
    return CSV.read(path, DataFrame)
end

# --- combine each pair ------------------------------------------------------

println("Building manuscript-ready combined tables ...")
println()

# --- 1. Primary analysis (full pooled N = 10000) ----------------------------
let marg = try_read(joinpath(DIR, "CA_paired_comparisons.csv")),
    real = try_read(joinpath(DIR, "CA_paired_comparisons_realized.csv"))
    if marg !== nothing && real !== nothing
        combined = combine_pair(marg, real; keys = [:comparison])
        # Reorder columns for readability:
        #   [comparison, n_marg/n_real, mean_marg/mean_real, ci_marg/ci_real, p, d]
        # The exact column set depends on `pairwise_paired_tests` output schema,
        # so just use whatever's there.
        round_cols!(combined)
        out = joinpath(DIR, "PAPER_TABLE_primary.csv")
        CSV.write(out, combined)
        println("✓ Primary (pooled N=10000)    -> $out  ($(nrow(combined)) rows)")
    end
end

# --- 2. Multi-replicate primary (10 × 1000) ---------------------------------
let marg = try_read(joinpath(DIR, "CA_paired_comparisons_multirep.csv")),
    real = try_read(joinpath(DIR, "CA_paired_comparisons_multirep_realized.csv"))
    if marg !== nothing && real !== nothing
        combined = combine_pair(marg, real; keys = [:comparison])
        round_cols!(combined)
        out = joinpath(DIR, "PAPER_TABLE_multirep.csv")
        CSV.write(out, combined)
        println("✓ Multi-replicate (10 reps)   -> $out  ($(nrow(combined)) rows)")
    end
end

# --- 3. Subgroup by stroke type ---------------------------------------------
let marg = try_read(joinpath(DIR, "CA_paired_comparisons_by_stroke_type.csv")),
    real = try_read(joinpath(DIR, "CA_paired_comparisons_by_stroke_type_realized.csv"))
    if marg !== nothing && real !== nothing
        # These have both `subgroup_column` and `subgroup_level` from
        # subgroup_paired_tests.
        keys = intersect([:comparison, :subgroup_column, :subgroup_level],
                          Symbol.(names(marg)),
                          Symbol.(names(real)))
        combined = combine_pair(marg, real; keys = collect(keys))
        round_cols!(combined)
        out = joinpath(DIR, "PAPER_TABLE_by_stroke_type.csv")
        CSV.write(out, combined)
        println("✓ Subgroup by stroke type     -> $out  ($(nrow(combined)) rows)")
    end
end

# --- 4. Subgroup by region (rurality) ---------------------------------------
let marg = try_read(joinpath(DIR, "CA_paired_comparisons_by_region.csv")),
    real = try_read(joinpath(DIR, "CA_paired_comparisons_by_region_realized.csv"))
    if marg !== nothing && real !== nothing
        keys = intersect([:comparison, :subgroup_column, :subgroup_level],
                          Symbol.(names(marg)),
                          Symbol.(names(real)))
        combined = combine_pair(marg, real; keys = collect(keys))
        round_cols!(combined)
        out = joinpath(DIR, "PAPER_TABLE_by_region.csv")
        CSV.write(out, combined)
        println("✓ Subgroup by region          -> $out  ($(nrow(combined)) rows)")
    end
end

println()
println("Done. The PAPER_TABLE_*.csv files have both reward versions side")
println("by side and are ready to paste into the manuscript's results tables.")
